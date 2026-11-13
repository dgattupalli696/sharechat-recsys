"""Stream test rows through one saved neural checkpoint and write a submission CSV.

Replays the original holdout first and requires the saved validation predictions
to be reproduced before predicting test. No training.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

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


def run(model_dir, preprocessing, data, output, cutoff, device="cuda", batch_size=4096):
    state = {**joblib.load(preprocessing),
             "category_remappings": joblib.load(model_dir / "category-remappings.joblib")}
    checkpoint = torch.load(model_dir / "best_model.pt", map_location="cpu", weights_only=True)
    model = HeterogeneousClassifier(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    def infer(path, validation):
        values, keys, labels, offset = [], [], [], 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8192):
            raw = pl.from_arrow(batch).with_columns(
                pl.Series("_row_id", np.arange(offset, offset + batch.num_rows, dtype=np.int64)))
            offset += batch.num_rows
            if validation:
                raw = raw.filter(pl.col("feature1") >= cutoff)
                if raw.is_empty():
                    continue
                keys.append(raw["_row_id"].to_numpy())
                labels.append(raw[TARGET].to_numpy())
            else:
                keys.append(raw["id"].to_numpy())
            x, c = transform_custom_features(features(raw.drop("_row_id"), expand_embeddings=True), state)
            values.append(predict(model, x, c, device, batch_size))
        return np.concatenate(values), np.concatenate(keys), np.concatenate(labels) if validation else None

    replay, row_ids, labels = infer(data / "train.parquet", True)
    with np.load(model_dir / "validation_predictions.npz") as saved:
        position = {row: i for i, row in enumerate(row_ids.tolist())}
        order = np.array([position[row] for row in saved["row_ids"].tolist()])
        np.testing.assert_array_equal(labels[order], saved["y_true"])
        error = float(np.max(np.abs(replay[order] - saved["probabilities"])))
    auc = float(roc_auc_score(labels, replay))
    if error > 5e-5:
        raise ValueError(f"Holdout replay differs from saved predictions: {error}")
    print(f"REPLAY: {len(labels):,} holdout rows, AUC {auc:.6f}, max error {error:.2e}", flush=True)
    template = pl.read_csv(data / "sample_submission.csv")
    probabilities, ids, _ = infer(data / "test.parquet", False)
    if not np.array_equal(ids, template["id"].to_numpy()):
        raise ValueError("Test IDs differ from the submission template")
    if not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("Invalid test probabilities")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "submission.csv"
    template.with_columns(pl.Series(TARGET, probabilities)).write_csv(path)
    report = {"holdout_replay_auc": auc, "holdout_rows": int(len(labels)), "max_replay_error": error,
              "file": path.name, "rows": int(len(probabilities)),
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "refit": False}
    (output / "prediction-results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--preprocessing", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cutoff", default="2026-10-28T08:06:52.397000")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run(args.model_dir, args.preprocessing, args.data_dir, args.output_dir,
        datetime.fromisoformat(args.cutoff), args.device)


if __name__ == "__main__":
    main()
