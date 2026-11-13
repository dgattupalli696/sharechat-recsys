import argparse
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import joblib
import numpy as np
import polars as pl
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score
import torch

from custom_network import transform_custom_features
from feature_models import HeterogeneousClassifier
from tabm_experiment import predict
from train import TARGET, features


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def combine(probabilities, weights):
    if set(probabilities) != set(weights) or abs(sum(weights.values()) - 1) > 1e-12:
        raise ValueError("Ensemble membership or weights differ")
    return sum(weight * probabilities[name].astype(np.float64) for name, weight in weights.items())


def run(bundle, data, output, *, device="cuda", batch_size=1024, worker_path=None):
    started = time.monotonic()
    manifest = json.loads((bundle / "manifest.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        if digest(bundle / name) != expected:
            raise ValueError(f"Model bundle checksum mismatch: {name}")
    versions = {name: version(name) for name in manifest["versions"]}
    if versions != manifest["versions"]:
        raise RuntimeError(f"Use the original training environment: expected {manifest['versions']}; got {versions}")
    torch.set_num_threads(4)
    state = joblib.load(bundle / "preprocessing.joblib")
    state = {**state, "category_remappings": joblib.load(bundle / "category-remappings.joblib")}
    models = {}
    for name, recorded in manifest["neural_models"].items():
        checkpoint = torch.load(bundle / recorded["checkpoint"], map_location="cpu", weights_only=True)
        if checkpoint["model_config"] != recorded["model_config"]:
            raise ValueError(f"Checkpoint configuration mismatch: {name}")
        model = HeterogeneousClassifier(**checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        models[name] = model
    columns = manifest["tree_columns"]
    vocabularies = json.loads((bundle / "category-vocabularies.json").read_text())
    category_expressions = [
        pl.col(name).replace_strict(
            {value: index for index, value in enumerate(values)}, default=-1, return_dtype=pl.Int32,
        )
        for name, values in vocabularies.items()
    ]
    output.mkdir(parents=True, exist_ok=True)
    cutoff = datetime.fromisoformat(manifest["cutoff"])
    template = pl.read_csv(data / "sample_submission.csv")
    if template.columns != ["id", TARGET] or template["id"].n_unique() != len(template):
        raise ValueError("Invalid sample-submission template")
    if pq.ParquetFile(data / "test.parquet").metadata.num_rows != len(template):
        raise ValueError("Test and sample-submission row counts differ")

    def infer(path, count, *, validation):
        predictions = {name: [] for name in models}
        identities, labels, offset, written = [], [], 0, 0
        with tempfile.TemporaryDirectory(prefix="finalist-inference-", dir=output) as directory:
            temporary = Path(directory)
            matrix = np.lib.format.open_memmap(
                temporary / "features.npy", mode="w+", dtype=np.float32,
                shape=(count, len(columns)),
            )
            for batch in pq.ParquetFile(path).iter_batches(batch_size=8192):
                raw = pl.from_arrow(batch).with_columns(
                    pl.Series("_source_row_id", np.arange(offset, offset + batch.num_rows)),
                )
                offset += len(raw)
                if validation:
                    raw = raw.filter(pl.col("feature1") >= cutoff)
                    if raw.is_empty():
                        continue
                    identities.append(raw["_source_row_id"].to_numpy())
                    labels.append(raw[TARGET].to_numpy())
                else:
                    if raw["id"].to_list() != template["id"].slice(written, len(raw)).to_list():
                        raise ValueError("Test IDs differ from the submission template")
                    identities.append(raw["id"].to_numpy())
                frame = features(raw.drop("_source_row_id"), expand_embeddings=True)
                if set(frame.columns) != set(columns) or len(frame.columns) != len(columns):
                    raise ValueError("Raw features differ from the training feature schema")
                frame = frame.select(columns)
                numeric, categorical = transform_custom_features(frame, state)
                for name, model in models.items():
                    predictions[name].append(predict(model, numeric, categorical, device, batch_size))
                encoded = frame.with_columns(category_expressions).select(
                    pl.all().cast(pl.Float32)
                ).to_numpy()
                if written + len(raw) > count:
                    raise ValueError("Prediction population exceeds the expected count")
                matrix[written:written + len(raw)] = encoded
                written += len(raw)
                if written % (8192 * 20) == 0:
                    print(f"{'Validation' if validation else 'Test'}: {written:,}/{count:,}", flush=True)
            if written != count:
                raise ValueError(f"Prediction population mismatch: {written} != {count}")
            matrix.flush()
            del matrix
            (temporary / "columns.json").write_text(json.dumps(columns))
            subprocess.run([
                sys.executable, str(worker_path or Path(__file__).with_name("lightgbm_worker.py")),
                "--predict-array", str(temporary / "features.npy"),
                "--model-file", str(bundle / "lightgbm.txt"),
                "--feature-columns", str(temporary / "columns.json"),
                "--output-dir", str(temporary), "--threads", "2",
            ], check=True)
            probabilities = {name: np.concatenate(parts) for name, parts in predictions.items()}
            probabilities["lightgbm"] = np.load(temporary / "probabilities.npy").copy()
        for name, values in probabilities.items():
            if len(values) != count or not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
                raise ValueError(f"Invalid probabilities: {name}")
        return probabilities, np.concatenate(identities), np.concatenate(labels) if validation else None

    validation, row_ids, labels = infer(
        data / "train.parquet", manifest["validation_rows"], validation=True,
    )
    with np.load(bundle / "validation-reference.npz") as reference:
        sorted_indices = np.argsort(row_ids)
        np.testing.assert_array_equal(np.sort(row_ids), np.sort(reference["row_ids"]))
        order = sorted_indices[np.searchsorted(row_ids[sorted_indices], reference["row_ids"])]
        np.testing.assert_array_equal(row_ids[order], reference["row_ids"])
        np.testing.assert_array_equal(labels[order], reference["y_true"])
        validation = {name: values[order] for name, values in validation.items()}
        validation["ensemble"] = combine(validation, manifest["ensemble_weights"])
        replay = {}
        for name, values in validation.items():
            expected = reference[name]
            error = float(np.max(np.abs(values - expected)))
            auc = float(roc_auc_score(reference["y_true"], values))
            expected_auc = float(roc_auc_score(reference["y_true"], expected))
            if error > 0.00005 or abs(auc - expected_auc) > 0.000005:
                raise ValueError(f"Validation replay mismatch for {name}: max error {error}, AUC {auc} != {expected_auc}")
            replay[name] = {"validation_auc": auc, "reference_auc": expected_auc, "max_probability_error": error}
        np.savez_compressed(
            output / "validation-replay.npz", y_true=reference["y_true"],
            row_ids=reference["row_ids"], **validation,
        )
    print(f"Full validation replay passed: {json.dumps(replay)}", flush=True)
    test, test_ids, _ = infer(data / "test.parquet", len(template), validation=False)
    np.testing.assert_array_equal(test_ids, template["id"].to_numpy())
    test["ensemble"] = combine(test, manifest["ensemble_weights"])
    submissions = {}
    for name, values in [("tabm", test["tabm32"]), ("ensemble", test["ensemble"])]:
        destination = output / f"submission-{name}.csv"
        template.with_columns(pl.Series(TARGET, values)).write_csv(destination)
        submissions[name] = {
            "file": destination.name, "sha256": digest(destination),
            "rows": len(values), "minimum": float(values.min()),
            "mean": float(values.mean()), "maximum": float(values.max()),
        }
    report = {
        "status": "COMPLETE", "refit": False, "automatic_competition_submission": False,
        "bundle_manifest_sha256": digest(bundle / "manifest.json"),
        "versions": versions, "validation_replay": replay, "submissions": submissions,
        "ensemble_weights": manifest["ensemble_weights"],
        "inference_seconds": time.monotonic() - started,
    }
    (output / "inference-results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    if args.batch_size <= 0 or (args.device == "cuda" and not torch.cuda.is_available()):
        parser.error("A positive batch size and the requested CUDA device are required")
    run(
        args.bundle, args.data_dir, args.output_dir, device=args.device,
        batch_size=args.batch_size, worker_path=globals().get("FROZEN_LIGHTGBM_WORKER_FILE"),
    )


if __name__ == "__main__":
    main()
