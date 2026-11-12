import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

from catboost import CatBoostClassifier
import numpy as np
import polars as pl
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

from train import TARGET, TIME, features


CUTOFF = "2026-10-28T08:06:52.397000"


def replay(path, model_path, identity_path, output, *, expected_auc, threads=2):
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    identities = pl.read_parquet(identity_path)
    row_ids = identities["_row_id"].to_numpy()
    parquet = pq.ParquetFile(path)
    rows = parquet.metadata.num_rows
    if len(np.unique(row_ids)) != len(row_ids) or (row_ids >= rows).any() or (row_ids < 0).any():
        raise ValueError("Invalid original validation row identities")
    positions = np.full(rows, -1, dtype=np.int64)
    positions[row_ids] = np.arange(len(row_ids))
    probabilities = np.empty(len(row_ids), dtype=np.float64)
    labels = np.empty(len(row_ids), dtype=np.uint8)
    cutoff = datetime.fromisoformat(CUTOFF)
    offset, written, valid_count = 0, 0, 0
    for batch in parquet.iter_batches(batch_size=8192):
        raw = pl.from_arrow(batch)
        if raw[TIME].null_count():
            raise ValueError("Missing source timestamps")
        is_valid = (raw[TIME] >= cutoff).to_numpy()
        valid_count += int(is_valid.sum())
        destination = positions[offset : offset + len(raw)]
        keep = destination >= 0
        if keep.any():
            if not is_valid[keep].all():
                raise ValueError("Reference identities contain a training row")
            selected = raw.filter(pl.Series(keep))
            frame = features(selected, expand_embeddings=False)
            if set(frame.columns) != set(model.feature_names_):
                raise ValueError("Summary features do not match the saved baseline model")
            frame = frame.select(model.feature_names_)
            probabilities[destination[keep]] = model.predict_proba(
                frame.to_pandas(), thread_count=threads,
            )[:, 1]
            labels[destination[keep]] = selected[TARGET].to_numpy()
            written += len(selected)
        offset += len(raw)
    if offset != rows or written != len(row_ids) or valid_count != len(row_ids):
        raise ValueError("Baseline replay did not cover the complete chronological holdout")
    if not np.isfinite(probabilities).all():
        raise ValueError("Nonfinite baseline probabilities")
    auc = float(roc_auc_score(labels, probabilities))
    if abs(auc - expected_auc) > 1e-8:
        raise ValueError(f"Original baseline AUC mismatch: {auc} versus {expected_auc}")
    np.savez_compressed(
        output / "validation_predictions.npz",
        y_true=labels, probabilities=probabilities, row_ids=row_ids,
    )
    with model_path.open("rb") as stream:
        model_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {
        "name": "B001-original-catboost", "validation_auc": auc,
        "training_rows": rows - valid_count, "validation_rows": valid_count,
        "cutoff": CUTOFF, "trees": model.tree_count_,
        "input_features": len(model.feature_names_),
        "model_path": str(model_path), "model_sha256": model_sha,
        "replay_seconds": time.monotonic() - started,
        "retraining": False, "competition_submission": False,
    }
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--identities", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-auc", type=float, required=True)
    args = parser.parse_args()
    replay(
        args.data, args.model, args.identities, args.output_dir,
        expected_auc=args.expected_auc,
    )


if __name__ == "__main__":
    main()
