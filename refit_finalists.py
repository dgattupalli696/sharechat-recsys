import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import joblib
import numpy as np
import polars as pl
import pyarrow.parquet as pq
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import torch
from torch.nn import functional as F

from custom_network import transform_custom_features
from feature_models import HeterogeneousClassifier
from relational_experiment import base_configuration
from tabm_experiment import predict
from train import features


WEIGHTS = {"tabm32": .45, "tabm8": .225, "dcn32": .225, "lightgbm": .1}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(path, cache, output):
    first = features(pl.from_arrow(next(pq.ParquetFile(path).iter_batches(batch_size=1))))
    numeric = [name for name, dtype in first.schema.items() if dtype.is_numeric()]
    categorical = [name for name, dtype in first.schema.items() if dtype == pl.String]
    vectors = [name for name, dtype in pl.scan_parquet(path).collect_schema().items() if isinstance(dtype, pl.List)]
    rows = pq.ParquetFile(path).metadata.num_rows
    matrix = np.lib.format.open_memmap(
        cache / "summary.npy", mode="w+", dtype=np.float32, shape=(rows, len(numeric)),
    )
    unique = [set() for _ in numeric]
    offset = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=8192):
        frame = features(pl.from_arrow(batch))
        array = np.array(frame.select(numeric).to_numpy(), dtype=np.float32)
        for i, seen in enumerate(unique):
            if seen is not None:
                seen.update(np.unique(array[~np.isnan(array[:, i]), i]).tolist())
                if np.isnan(array[:, i]).any():
                    seen.add(None)
                if len(seen) > 16:
                    unique[i] = None
        array[~np.isfinite(array)] = np.nan
        matrix[offset:offset + len(array)] = array
        offset += len(array)
    if offset != rows:
        raise ValueError("Full training population changed")
    continuous = [i for i, seen in enumerate(unique) if seen is None]
    matrix.flush()
    print(f"PREPARE: fitting medians on all {rows:,} rows", flush=True)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True, add_indicator=True, copy=False)
    imputer.fit(matrix)
    scaler = StandardScaler(copy=False)
    for start in range(0, rows, 8192):
        scaler.partial_fit(imputer.transform(np.array(matrix[start:start + 8192], copy=True)))
    del matrix
    (cache / "summary.npy").unlink()
    gc.collect()
    vocabularies, counts = {}, []
    for name in categorical:
        table = (
            pl.scan_parquet(path).select(pl.col(name).fill_null("__MISSING__"))
            .group_by(name, maintain_order=True).len().collect()
        )
        vocabulary = table[name].to_list()
        if len(vocabulary) >= 2 ** 24:
            raise ValueError("Category codes cannot be represented exactly as float32")
        vocabularies[name] = vocabulary
        mapping = dict(zip(vocabulary, table["len"].to_list()))
        counts.append(np.array([0, *[mapping[value] for value in sorted(vocabulary)]], dtype=np.int64))
    categories = [sorted(vocabularies[name]) for name in categorical]
    encoder = OrdinalEncoder(
        categories=categories, handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64,
    )
    encoder.fit(np.array([[values[0] for values in categories]], dtype=object))
    remappings, cardinalities = [], []
    for count in counts:
        retained = np.flatnonzero((count >= 10) & (np.arange(len(count)) != 0))
        if len(count) > 1000:
            retained = np.empty(0, dtype=np.int64)
        mapping = np.zeros(len(count), dtype=np.int64)
        mapping[retained] = np.arange(1, len(retained) + 1)
        remappings.append(mapping)
        cardinalities.append(len(retained) + 1)
    state = {
        "numeric_columns": numeric, "categorical_columns": categorical,
        "imputer": imputer, "scaler": scaler, "encoder": encoder,
        "vector_fields": vectors, "n_dense": len(numeric) + len(imputer.indicator_.features_),
        "original_cardinalities": [len(count) for count in counts],
        "cardinalities": cardinalities, "category_remappings": remappings,
        "min_category_count": 10,
    }
    joblib.dump(state, output / "preprocessing.joblib")
    (output / "category-vocabularies.json").write_text(json.dumps(vocabularies) + "\n")
    width = state["n_dense"] + 34 * len(vectors)
    numeric_array = np.lib.format.open_memmap(
        cache / "numeric.npy", mode="w+", dtype=np.float32, shape=(rows, width),
    )
    category_array = np.lib.format.open_memmap(
        cache / "categorical.npy", mode="w+", dtype=np.int64, shape=(rows, len(categorical)),
    )
    offset = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=8192):
        frame = features(pl.from_arrow(batch), expand_embeddings=True)
        x, c = transform_custom_features(frame, state)
        numeric_array[offset:offset + len(frame)] = x.numpy()
        category_array[offset:offset + len(frame)] = c.numpy()
        offset += len(frame)
    numeric_array.flush()
    category_array.flush()
    metadata = pl.scan_parquet(path).select("feature1", "watched").with_row_index("_row_id").collect()
    if metadata["feature1"].null_count() or set(metadata["watched"].unique().to_list()) != {0, 1}:
        raise ValueError("Full training requires nonmissing timestamps and both target classes")
    labels = metadata["watched"].to_numpy().copy()
    order = metadata.sort("feature1")["_row_id"].to_numpy().astype(np.int64)
    data = {
        "x_train": torch.from_numpy(numeric_array), "state": state,
        "protocol": {"continuous_indices": continuous},
        "y_train": torch.tensor(labels, dtype=torch.float32),
    }
    config = base_configuration(data, cardinalities)
    print(f"PREPARED: {rows:,} rows; numeric shape {numeric_array.shape}", flush=True)
    return numeric_array, category_array, labels, order, state, config


def train_fixed(model, config, numeric, categorical, labels, order, steps, destination, device):
    if steps <= 0 or len(numeric) != len(labels) or len(order) != len(labels):
        raise ValueError("Invalid full-training budget or population")
    if steps < math.ceil(len(labels) / 1024) or not np.array_equal(np.sort(order), np.arange(len(labels))):
        raise ValueError("Every labeled row must be visited in a complete first epoch")
    destination.mkdir()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0005, weight_decay=.01)
    generator = torch.Generator().manual_seed(42)
    x, c = torch.from_numpy(numeric), torch.from_numpy(categorical)
    y = torch.tensor(labels, dtype=torch.float32)
    chronological = torch.from_numpy(order)
    completed, epoch, history = 0, 0, []
    started = time.monotonic()
    while completed < steps:
        epoch += 1
        model.train()
        indices = chronological[torch.randperm(len(labels), generator=generator)]
        loss_sum, seen = 0., 0
        for batch in indices.split(1024):
            optimizer.zero_grad(set_to_none=True)
            logits = model(x[batch].to(device), c[batch].to(device))
            targets = y[batch].to(device).unsqueeze(1).expand_as(logits)
            loss = F.binary_cross_entropy_with_logits(logits, targets)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite full-training loss")
            loss.backward()
            optimizer.step()
            completed += 1
            seen += len(batch)
            loss_sum += float(loss.item()) * len(batch)
            if completed == steps:
                break
        record = {"epoch": epoch, "optimizer_steps": completed, "training_loss": loss_sum / seen}
        history.append(record)
        print(json.dumps(record), flush=True)
    replay_before = predict(model, x[:16], c[:16], device, 16)
    torch.save({
        "model_config": config,
        "state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
    }, destination / "model.pt")
    checkpoint = torch.load(destination / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    np.testing.assert_array_equal(replay_before, predict(model, x[:16], c[:16], device, 16))
    result = {
        "training_rows": len(labels), "optimizer_steps": completed,
        "selection": "validation-frozen fractional-epoch budget; no refit validation",
        "history": history, "training_seconds": time.monotonic() - started,
        "checkpoint_sha256": digest(destination / "model.pt"),
    }
    (destination / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def tree_matrix(path, destination, vocabularies, columns=None):
    rows = pq.ParquetFile(path).metadata.num_rows
    if columns is None:
        columns = features(pl.from_arrow(next(pq.ParquetFile(path).iter_batches(batch_size=1))), True).columns
    matrix = np.lib.format.open_memmap(
        destination, mode="w+", dtype=np.float32, shape=(rows, len(columns)),
    )
    expressions = [
        pl.col(name).replace_strict(
            {value: i for i, value in enumerate(values)}, default=-1, return_dtype=pl.Int32,
        )
        for name, values in vocabularies.items()
    ]
    offset = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=8192):
        frame = features(pl.from_arrow(batch), True).select(columns)
        matrix[offset:offset + len(frame)] = frame.with_columns(expressions).select(
            pl.all().cast(pl.Float32),
        ).to_numpy()
        offset += len(frame)
    if offset != rows:
        raise ValueError("Tree feature population changed")
    matrix.flush()
    return columns


def run(data, output, selected, device="cuda", worker=None):
    tabm_only = selected.get("tabm_only", False)
    if not tabm_only and selected["ensemble_weights"] != WEIGHTS:
        raise ValueError("Refit weights differ from the selected T016 ensemble")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    started = time.monotonic()
    template = pl.read_csv(data / "sample_submission.csv")
    if template.columns != ["id", "watched"] or template["id"].n_unique() != len(template):
        raise ValueError("Invalid test template")
    if pq.ParquetFile(data / "test.parquet").metadata.num_rows != len(template):
        raise ValueError("Test population differs from the submission template")
    neural_results = {}
    with tempfile.TemporaryDirectory(prefix="full-refit-") as directory:
        cache = Path(directory)
        numeric, categorical, labels, order, state, base = prepare(data / "train.parquet", cache, output)
        if len(labels) != selected["full_training_rows"]:
            raise ValueError("Refit requires every labeled row")
        steps_per_epoch = math.ceil(len(labels) / 1024)
        for name, specification in selected["neural"].items():
            torch.manual_seed(42)
            config = {
                **base, **specification.get("model_config", {}),
                "members": specification["members"], "architecture": specification["architecture"],
            }
            steps = round(specification["fractional_epochs"] * steps_per_epoch)
            model = HeterogeneousClassifier(**config).to(device)
            print(f"START FULL {name}: {len(labels):,} rows, {steps} fixed steps", flush=True)
            neural_results[name] = train_fixed(
                model, config, numeric, categorical, labels, order, steps, output / name, device,
            )
            del model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
        del numeric, categorical
        (cache / "numeric.npy").unlink()
        (cache / "categorical.npy").unlink()
        gc.collect()
        models, predictions = {}, {}
        for name in selected["neural"]:
            checkpoint = torch.load(output / name / "model.pt", map_location="cpu", weights_only=True)
            model = HeterogeneousClassifier(**checkpoint["model_config"]).to(device)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            models[name] = model
            predictions[name] = np.empty(len(template), dtype=np.float32)
        offset = 0
        for batch in pq.ParquetFile(data / "test.parquet").iter_batches(batch_size=8192):
            raw = pl.from_arrow(batch)
            if raw["id"].to_list() != template["id"].slice(offset, len(raw)).to_list():
                raise ValueError("Test IDs differ from the template")
            x, c = transform_custom_features(features(raw, True), state)
            for name, model in models.items():
                predictions[name][offset:offset + len(raw)] = predict(model, x, c, device, 1024)
            offset += len(raw)
        if offset != len(template):
            raise ValueError("Incomplete neural test predictions")
        del models, model, checkpoint, x, c
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        if not tabm_only:
            vocabularies = json.loads((output / "category-vocabularies.json").read_text())
            columns = tree_matrix(data / "train.parquet", cache / "train.npy", vocabularies)
            (cache / "metadata.json").write_text(json.dumps({"columns": columns, "categories": list(vocabularies)}))
            np.savez(cache / "labels.npz", train=labels)
            tree_output = output / "lightgbm"
            tree_output.mkdir()
            worker = worker or Path(__file__).with_name("lightgbm_worker.py")
            subprocess.run([
                sys.executable, str(worker), "--array-dir", str(cache), "--output-dir", str(tree_output),
                "--iterations", str(selected["lightgbm_trees"]), "--threads", "4", "--fixed-budget", "--full-training",
            ], check=True)
            (cache / "train.npy").unlink()
            tree_matrix(data / "test.parquet", cache / "test.npy", vocabularies, columns)
            (cache / "columns.json").write_text(json.dumps(columns))
            subprocess.run([
                sys.executable, str(worker), "--predict-array", str(cache / "test.npy"),
                "--model-file", str(tree_output / "model.txt"),
                "--feature-columns", str(cache / "columns.json"),
                "--output-dir", str(tree_output), "--threads", "4",
            ], check=True)
            predictions["lightgbm"] = np.load(tree_output / "probabilities.npy")
    for name, values in predictions.items():
        if len(values) != len(template) or not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
            raise ValueError(f"Invalid full-refit predictions: {name}")
    np.savez_compressed(output / "test-component-predictions.npz", **predictions)
    outputs = [("tabm", predictions["tabm32"])]
    if not tabm_only:
        outputs.append(("ensemble", sum(WEIGHTS[name] * values.astype(np.float64) for name, values in predictions.items())))
    submissions = {}
    for name, values in outputs:
        path = output / f"submission-full-{name}.csv"
        template.with_columns(pl.Series("watched", values)).write_csv(path)
        submissions[name] = {"file": path.name, "sha256": digest(path), "rows": len(values)}
    report = {
        "status": "COMPLETE", "full_training_rows": len(labels),
        "validation_auc": None, "validation_used_for_refit_selection": False,
        "selection_reference": selected.get(
            "selection_reference", "T016 holdout AUC 0.8408057533; not an evaluation of this full-data refit",
        ),
        "neural": neural_results, "ensemble_weights": None if tabm_only else WEIGHTS,
        "preprocessing_training_rows": int(state["scaler"].n_samples_seen_),
        "preprocessing_sha256": digest(output / "preprocessing.joblib"),
        "category_vocabulary_sha256": digest(output / "category-vocabularies.json"),
        "training_labels_sha256": hashlib.sha256(labels.astype(np.uint8).tobytes()).hexdigest(),
        "lightgbm": None if tabm_only else {
            **json.loads((tree_output / "worker-results.json").read_text()),
            "checkpoint_sha256": digest(tree_output / "model.txt"),
        },
        "submissions": submissions, "competition_submission": False,
        "seconds": time.monotonic() - started,
    }
    (output / "refit-results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    run(args.data, args.output, json.loads(args.selected.read_text()), args.device)


if __name__ == "__main__":
    main()
