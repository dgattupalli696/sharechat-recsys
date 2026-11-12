import argparse
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

import joblib
import numpy as np
import polars as pl
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score
import torch

from custom_network import RecommendationNetwork, transform_custom_features
from tabm_experiment import predict
from train import TARGET, features


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cuda")
    args = parser.parse_args()
    if min(args.batch_size, args.threads) <= 0:
        parser.error("Batch size and threads must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA inference requested but unavailable")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS inference requested but unavailable")
    started = time.monotonic()
    torch.set_num_threads(args.threads)
    training = json.loads((args.model_dir / "results.json").read_text())
    variants = [entry for entry in training["results"] if entry["variant"] == "attention"]
    if len(variants) != 1:
        raise ValueError("Expected one recorded attention checkpoint")
    reference = variants[0]
    versions = {name: version(name) for name in training["versions"]}
    if versions != training["versions"]:
        raise RuntimeError(
            "Inference package versions differ from training; use the original Kaggle "
            f"environment. Expected {training['versions']}; got {versions}"
        )
    checkpoint_path = args.model_dir / "attention/best_model.pt"
    preprocessor_path = args.model_dir / "preprocessing.joblib"
    state = joblib.load(preprocessor_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["model_config"] != reference["model_config"]:
        raise ValueError("Checkpoint model configuration differs from the training record")
    model = RecommendationNetwork(**checkpoint["model_config"]).to(args.device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    print(
        f"Inference device: {args.device}; unchanged best epoch: {reference['best_epoch']}; "
        f"training rows: {training['training_rows']:,}; no retraining",
        flush=True,
    )

    def probabilities(raw):
        numeric, categorical = transform_custom_features(
            features(raw, expand_embeddings=True), state
        )
        return predict(model, numeric, categorical, args.device, args.batch_size)

    cutoff = datetime.fromisoformat(training["cutoff"])
    labels, validation_predictions = [], []
    for batch in pq.ParquetFile(args.data_dir / "train.parquet").iter_batches(
        batch_size=8 * args.batch_size
    ):
        raw = pl.from_arrow(batch).filter(pl.col("feature1") >= cutoff)
        if raw.is_empty():
            continue
        labels.append(raw[TARGET].to_numpy())
        validation_predictions.append(probabilities(raw))
    if not labels:
        raise ValueError("No validation rows found for the recorded timestamp cutoff")
    labels = np.concatenate(labels)
    validation_predictions = np.concatenate(validation_predictions)
    if len(labels) != training["validation_rows"]:
        raise ValueError("Replayed validation row count differs from the training record")
    validation_auc = float(roc_auc_score(labels, validation_predictions))
    if abs(validation_auc - reference["validation_auc"]) > 0.000005:
        raise ValueError(
            f"Replayed AUC {validation_auc} differs from checkpoint AUC "
            f"{reference['validation_auc']}"
        )
    print(f"Replayed validation ROC AUC: {validation_auc:.9f}", flush=True)

    template = pl.read_csv(args.data_dir / "sample_submission.csv")
    if template.columns != ["id", TARGET] or template["id"].n_unique() != template.height:
        raise ValueError("Unexpected submission columns or duplicate template IDs")
    test = pq.ParquetFile(args.data_dir / "test.parquet")
    if test.metadata.num_rows != template.height:
        raise ValueError("Test and template row counts differ")
    predictions, offset = [], 0
    for batch in test.iter_batches(batch_size=8 * args.batch_size):
        raw = pl.from_arrow(batch)
        if "id" not in raw.columns:
            raise ValueError("Test data is missing submission IDs")
        if raw["id"].to_list() != template["id"].slice(offset, raw.height).to_list():
            raise ValueError("Test IDs do not match sample submission order")
        predictions.append(probabilities(raw))
        offset += raw.height
        if offset % (25 * 8 * args.batch_size) == 0:
            print(f"Predicted {offset:,}/{template.height:,} test rows", flush=True)
    predictions = np.concatenate(predictions)
    if len(predictions) != template.height or not np.isfinite(predictions).all():
        raise ValueError("Invalid prediction count or nonfinite probabilities")
    if not ((predictions >= 0) & (predictions <= 1)).all():
        raise ValueError("Probabilities must be in [0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "submission.csv"
    template.with_columns(pl.Series(TARGET, predictions)).write_csv(destination)
    report = {
        "checkpoint_best_epoch": reference["best_epoch"],
        "training_rows": training["training_rows"],
        "validation_rows": len(labels),
        "training_validation_auc": reference["validation_auc"],
        "replayed_validation_auc": validation_auc,
        "test_rows": len(predictions),
        "prediction_min": float(predictions.min()),
        "prediction_mean": float(predictions.mean()),
        "prediction_max": float(predictions.max()),
        "checkpoint_sha256": digest(checkpoint_path),
        "preprocessor_sha256": digest(preprocessor_path),
        "training_results_sha256": digest(args.model_dir / "results.json"),
        "submission_sha256": digest(destination),
        "submission_bytes": destination.stat().st_size,
        "device": args.device,
        "batch_size": args.batch_size,
        "versions": versions,
        "inference_seconds": time.monotonic() - started,
        "refit": False,
        "automatic_competition_submission": False,
    }
    (args.output_dir / "inference-results.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        f"Saved {len(predictions):,} predictions; no retraining or automatic submission",
        flush=True,
    )


if __name__ == "__main__":
    main()
