from datetime import datetime
import gc
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from catboost import CatBoost, Pool
import joblib
import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
from sklearn.decomposition import PCA
from sklearn.metrics import log_loss, roc_auc_score
import torch

from custom_network import pack_vectors
from feature_models import pairwise_vector_features
from train import TARGET, load_sample, model


def cross_frame(frame, fields):
    pairs = torch.triu_indices(len(fields), len(fields), offset=1)
    width = pairs.shape[1] * 4
    values = np.empty((len(frame), width), dtype=np.float32)
    for start in range(0, len(frame), 8192):
        packed, _ = pack_vectors(frame.slice(start, 8192), fields)
        values[start : start + len(packed)] = pairwise_vector_features(packed, pairs).numpy()
    return pd.DataFrame(values, columns=[f"vector_cross_{i}" for i in range(width)])


def matrices(train, valid, fields, kind, output):
    columns = [name for name in train.columns if name not in {TARGET, "_time", "_row_id"}]
    categories = [name for name in columns if train.schema[name] == pl.String]
    if kind in {"catboost-expanded", "lightgbm"}:
        return train.select(columns).to_pandas(), valid.select(columns).to_pandas(), categories, []
    summary = [name for name in columns if "_dim_" not in name]
    x_train, x_valid = train.select(summary).to_pandas(), valid.select(summary).to_pandas()
    embeddings = []
    if kind == "catboost-crosses":
        return (
            pd.concat([x_train, cross_frame(train, fields)], axis=1),
            pd.concat([x_valid, cross_frame(valid, fields)], axis=1),
            categories, embeddings,
        )
    transforms = {}
    additions_train, additions_valid = {}, {}
    components = int(kind.rsplit("-", 1)[1]) if kind.startswith("pca-") else None
    for field in fields:
        dimensions = [f"{field}_dim_{i}" for i in range(32)]
        raw_train = np.array(train.select(dimensions).to_numpy(), dtype=np.float32, order="C")
        raw_valid = np.array(valid.select(dimensions).to_numpy(), dtype=np.float32, order="C")
        present_train, present_valid = np.isfinite(raw_train).all(1), np.isfinite(raw_valid).all(1)
        raw_train[~present_train], raw_valid[~present_valid] = 0, 0
        additions_train[field + "_present"] = present_train.astype(np.float32)
        additions_valid[field + "_present"] = present_valid.astype(np.float32)
        if components is not None:
            if present_train.sum() < components:
                raise ValueError(f"{field} has insufficient training vectors for PCA-{components}")
            transform = PCA(n_components=components, svd_solver="full")
            transform.fit(raw_train[present_train])
            encoded_train, encoded_valid = transform.transform(raw_train), transform.transform(raw_valid)
            encoded_train[~present_train], encoded_valid[~present_valid] = 0, 0
            for index in range(components):
                name = f"{field}_pca_{index}"
                additions_train[name], additions_valid[name] = encoded_train[:, index], encoded_valid[:, index]
            transforms[field] = transform
        else:
            additions_train[field], additions_valid[field] = list(raw_train), list(raw_valid)
            embeddings.append(field)
    if transforms:
        joblib.dump(transforms, output / "pca.joblib")
    return (
        pd.concat([x_train, pd.DataFrame(additions_train)], axis=1),
        pd.concat([x_valid, pd.DataFrame(additions_valid)], axis=1),
        categories, embeddings,
    )


def run_trees(
    path, output, cutoff, *, device="cuda", iterations=400,
    light_iterations=1000, only=None, worker_path=None,
    valid_end=None, fixed_budget=False,
):
    rows = pq.ParquetFile(path).metadata.num_rows
    sample = load_sample(path, rows, 42, expand_embeddings=True).with_row_index("_row_id").sort("_time")
    split = int(sample["_time"].search_sorted(datetime.fromisoformat(cutoff), side="left"))
    if valid_end:
        end = int(sample["_time"].search_sorted(datetime.fromisoformat(valid_end), side="left"))
        sample = sample.slice(0, end)
    train, valid = sample.slice(0, split), sample.slice(split)
    fields = sorted({
        name.rsplit("_dim_", 1)[0] for name in sample.columns if "_dim_" in name
    }, key=lambda value: int(value.removeprefix("feature")))
    y_train, y_valid = train[TARGET].to_numpy().copy(), valid[TARGET].to_numpy().copy()
    if rows == 1537638 and cutoff == "2026-10-28T08:06:52.397000" and valid_end is None:
        assert len(y_train) == 1230110 and len(y_valid) == 307528
    report = {
        "stage": "trees", "training_rows": len(y_train), "validation_rows": len(y_valid),
        "cutoff": cutoff, "results": [], "competition_submission": False,
    }
    for kind in [
        "catboost-expanded", "pca-8", "pca-16", "catboost-crosses",
        "catboost-native-lda", "lightgbm",
    ]:
        if only is not None and kind not in only:
            continue
        destination = output / kind
        destination.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        x_train, x_valid, categories, embeddings = matrices(
            train, valid, fields, kind, destination
        )
        preparation_seconds = time.monotonic() - started
        input_features = x_train.shape[1]
        print(f"START {kind}: {len(y_train)} train / {len(y_valid)} holdout", flush=True)
        fit_started = time.monotonic()
        if kind == "lightgbm":
            with tempfile.TemporaryDirectory() as directory:
                temporary = Path(directory)
                x_train.to_parquet(temporary / "train.parquet", index=False)
                x_valid.to_parquet(temporary / "valid.parquet", index=False)
                np.savez(temporary / "labels.npz", train=y_train, valid=y_valid)
                (temporary / "categories.json").write_text(json.dumps(categories))
                del x_train, x_valid
                gc.collect()
                subprocess.run([
                    sys.executable, str(worker_path or Path(__file__).with_name("lightgbm_worker.py")),
                    "--input-dir", str(temporary), "--output-dir", str(destination),
                    "--iterations", str(light_iterations),
                    *(["--fixed-budget"] if fixed_budget else []),
                ], check=True)
            arrays = np.load(destination / "validation_predictions.npz")
            probabilities = arrays["probabilities"]
            worker = json.loads((destination / "worker-results.json").read_text())
            best_iteration = worker["best_iteration"]
            backend, library_version = worker["backend"], worker["library_version"]
        else:
            native = kind == "catboost-native-lda"
            backend = "CPU" if native or device == "cpu" else "GPU"
            if native:
                estimator = CatBoost({
                    "iterations": iterations, "depth": 6, "learning_rate": 0.06,
                    "loss_function": "Logloss", "eval_metric": "AUC",
                    "random_seed": 42, "thread_count": 4, "task_type": backend,
                    "max_ctr_complexity": 1, "allow_writing_files": False, "verbose": 50,
                    "embedding_processing": {"embedding_processing": {"default": ["LDA"]}},
                })
            else:
                estimator = model(iterations, 4, backend)
            training_pool = Pool(
                x_train, label=y_train, cat_features=categories, embedding_features=embeddings
            )
            validation_pool = Pool(
                x_valid, label=y_valid, cat_features=categories, embedding_features=embeddings
            )
            if fixed_budget:
                estimator.fit(training_pool)
            else:
                estimator.fit(
                    training_pool, eval_set=validation_pool, early_stopping_rounds=60
                )
            probabilities = estimator.predict(validation_pool, prediction_type="Probability")[:, 1]
            estimator.save_model(str(destination / "model.cbm"))
            best_iteration = estimator.tree_count_
            import catboost

            library_version = catboost.__version__
            del training_pool, validation_pool
        probabilities = np.asarray(probabilities)
        if not np.isfinite(probabilities).all():
            raise ValueError(f"Nonfinite {kind} predictions")
        np.savez_compressed(
            destination / "validation_predictions.npz",
            y_true=y_valid, probabilities=probabilities,
            row_ids=valid["_row_id"].to_numpy(),
        )
        result = {
            "name": kind, "validation_auc": float(roc_auc_score(y_valid, probabilities)),
            "validation_logloss": float(log_loss(y_valid, probabilities, labels=[0, 1])),
            "preparation_seconds": preparation_seconds,
            "training_and_prediction_seconds": time.monotonic() - fit_started,
            "best_iteration": best_iteration, "backend": backend,
            "library_version": library_version, "input_features": input_features,
        }
        report["results"].append(result)
        (destination / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"RESULT {kind}: AUC={result['validation_auc']:.10f}", flush=True)
        if kind != "lightgbm":
            del estimator, x_train, x_valid
        gc.collect()
    report["status"] = "COMPLETE"
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
