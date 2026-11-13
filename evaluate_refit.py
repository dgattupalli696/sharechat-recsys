import argparse
from datetime import datetime
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import joblib
import numpy as np
import polars as pl
import pyarrow.parquet as pq
from sklearn.metrics import log_loss, roc_auc_score
import torch

from custom_network import transform_custom_features
from feature_models import HeterogeneousClassifier
from refit_finalists import WEIGHTS, digest, tree_matrix
from tabm_experiment import predict
from train import features


def evaluate(bundle, source, output, *, device="cuda", worker=None):
    manifest = json.loads((bundle / "manifest.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        if digest(bundle / name) != expected:
            raise ValueError(f"Checkpoint bundle checksum mismatch: {name}")
    actual_versions = {name: version(name) for name in manifest["versions"]}
    if actual_versions != manifest["versions"]:
        raise RuntimeError("Checkpoint evaluation requires the original training environment")
    if manifest["ensemble_weights"] != WEIGHTS:
        raise ValueError("Ensemble weights differ from the frozen recipe")
    torch.set_num_threads(4)
    state = joblib.load(bundle / "preprocessing.joblib")
    models = {}
    for name in ["tabm32", "tabm8", "dcn32"]:
        checkpoint = torch.load(bundle / f"{name}.pt", map_location="cpu", weights_only=True)
        model = HeterogeneousClassifier(**checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        models[name] = model
    with np.load(bundle / "evaluation-reference.npz") as reference:
        expected_ids = reference["row_ids"].copy()
        expected_labels = reference["y_true"].copy()
    if len(np.unique(expected_ids)) != len(expected_ids):
        raise ValueError("Evaluation requires unique original row identities")
    output.mkdir(parents=True, exist_ok=True)
    cutoff = datetime.fromisoformat(manifest["cutoff"])
    with tempfile.TemporaryDirectory(prefix="checkpoint-sanity-") as directory:
        temporary = Path(directory)
        parquet = temporary / "old-holdout.parquet"
        (
            pl.scan_parquet(source).with_row_index("id")
            .filter(pl.col("feature1") >= cutoff).sink_parquet(parquet)
        )
        if pq.ParquetFile(parquet).metadata.num_rows != len(expected_ids):
            raise ValueError("Former holdout population differs from the original split")
        probabilities = {name: [] for name in models}
        identities, labels = [], []
        for batch in pq.ParquetFile(parquet).iter_batches(batch_size=8192):
            raw = pl.from_arrow(batch)
            identities.append(raw["id"].to_numpy())
            labels.append(raw["watched"].to_numpy())
            numeric, categorical = transform_custom_features(features(raw, True), state)
            for name, model in models.items():
                probabilities[name].append(predict(model, numeric, categorical, device, 1024))
        probabilities = {name: np.concatenate(parts) for name, parts in probabilities.items()}
        vocabularies = json.loads((bundle / "category-vocabularies.json").read_text())
        columns = tree_matrix(parquet, temporary / "features.npy", vocabularies)
        (temporary / "columns.json").write_text(json.dumps(columns))
        subprocess.run([
            sys.executable, str(worker or Path(__file__).with_name("lightgbm_worker.py")),
            "--predict-array", str(temporary / "features.npy"),
            "--model-file", str(bundle / "lightgbm.txt"),
            "--feature-columns", str(temporary / "columns.json"),
            "--output-dir", str(temporary), "--threads", "2",
        ], check=True)
        probabilities["lightgbm"] = np.load(temporary / "probabilities.npy").copy()
    ids, target = np.concatenate(identities), np.concatenate(labels)
    sorted_indices = np.argsort(ids)
    np.testing.assert_array_equal(np.sort(ids), np.sort(expected_ids))
    order = sorted_indices[np.searchsorted(ids[sorted_indices], expected_ids)]
    np.testing.assert_array_equal(ids[order], expected_ids)
    np.testing.assert_array_equal(target[order], expected_labels)
    probabilities = {name: values[order] for name, values in probabilities.items()}
    probabilities["ensemble"] = sum(
        weight * probabilities[name].astype(np.float64) for name, weight in WEIGHTS.items()
    )
    scores = {}
    for name, values in probabilities.items():
        if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
            raise ValueError(f"Invalid checkpoint predictions: {name}")
        scores[name] = {
            "in_sample_auc": float(roc_auc_score(expected_labels, values)),
            "in_sample_logloss": float(log_loss(expected_labels, values, labels=[0, 1])),
        }
    np.savez_compressed(
        output / "former-holdout-predictions.npz",
        row_ids=expected_ids, y_true=expected_labels, **probabilities,
    )
    report = {
        "status": "COMPLETE", "evaluation": "in-sample sanity check on former holdout",
        "evaluation_rows": len(expected_ids), "training_overlap_rows": len(expected_ids),
        "checkpoint_training_rows": manifest["checkpoint_training_rows"],
        "unbiased_validation": False, "used_for_model_selection": False,
        "additional_training": False, "competition_submission": False,
        "bundle_manifest_sha256": digest(bundle / "manifest.json"),
        "scores": scores, "ensemble_weights": WEIGHTS,
        "interpretation": "Every evaluated row was part of the full-data refit; these scores are not comparable to the original out-of-sample holdout AUCs.",
    }
    (output / "sanity-results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    evaluate(args.bundle, args.source, args.output, device=args.device)


if __name__ == "__main__":
    main()
