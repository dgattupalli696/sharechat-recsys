import argparse
import gc
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import time

import joblib
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch

from custom_network import prepare_custom_features
from tabm_experiment import TabMClassifier, fit_model_steps
from train import ROOT, TARGET, load_sample


PRIMARY_CUTOFF = "2026-10-28T08:06:52.397000"
SECONDARY_CUTOFF = "2026-10-27T08:06:52.397000"
SECONDARY_END = "2026-10-27T18:00:00"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_data(path, output, *, secondary=False):
    from datetime import datetime

    started = time.monotonic()
    total = pq.ParquetFile(path).metadata.num_rows
    sample = (
        load_sample(path, total, 42, expand_embeddings=True)
        .with_row_index("_row_id").sort("_time")
    )
    if sample["_time"].null_count():
        raise ValueError("Missing timestamps in experiment input")
    cutoff = datetime.fromisoformat(SECONDARY_CUTOFF if secondary else PRIMARY_CUTOFF)
    split = int(sample["_time"].search_sorted(cutoff, side="left"))
    if secondary:
        end = int(sample["_time"].search_sorted(
            datetime.fromisoformat(SECONDARY_END), side="left"
        ))
        sample = sample.slice(0, end)
    train, valid = sample.slice(0, split), sample.slice(split)
    validation_row_ids = valid["_row_id"].to_numpy().copy()
    train, valid = train.drop("_row_id"), valid.drop("_row_id")
    if train.is_empty() or valid.is_empty():
        raise ValueError("The fixed temporal cutoff does not split this dataset")
    for name, frame in [("training", train), ("validation", valid)]:
        if set(frame[TARGET].unique().to_list()) != {0, 1}:
            raise ValueError(f"{name} requires both binary classes")
    x_train, c_train, x_valid, c_valid, state = prepare_custom_features(train, valid, 10)
    unique_counts = train.select(pl.col(state["numeric_columns"]).n_unique()).row(0)
    continuous = [i for i, count in enumerate(unique_counts) if count > 16]
    original_train, original_valid = [], []
    category_counts = []
    original_train_array = state["encoder"].transform(
        train.select(state["categorical_columns"]).fill_null("__MISSING__").to_numpy()
    ).astype(np.int64) + 1
    original_valid_array = state["encoder"].transform(
        valid.select(state["categorical_columns"]).fill_null("__MISSING__").to_numpy()
    ).astype(np.int64) + 1
    for index, cardinality in enumerate(state["original_cardinalities"]):
        original_train.append(original_train_array[:, index].copy())
        original_valid.append(original_valid_array[:, index].copy())
        category_counts.append(np.bincount(original_train[-1], minlength=cardinality))
    # Keep identifiers outside model inputs for paired/group-aware evaluation.
    identity = (
        pl.scan_parquet(path).select("request_id", "feature1")
        .with_row_index("_row_id").collect()[validation_row_ids]
    )
    identity.write_parquet(output / "validation-identities.parquet")
    protocol = {
        "source_rows": total, "training_rows": train.height, "validation_rows": valid.height,
        "cutoff": cutoff.isoformat(), "secondary": secondary,
        "secondary_end": SECONDARY_END if secondary else None,
        "validation_labels_sha256": hashlib.sha256(
            valid[TARGET].to_numpy().astype(np.uint8).tobytes()
        ).hexdigest(),
        "validation_identities_sha256": sha256(output / "validation-identities.parquet"),
        "continuous_indices": continuous,
        "versions": {
            name: version(name)
            for name in ["torch", "tabm", "rtdl_num_embeddings", "numpy", "polars", "scikit-learn"]
        },
        "preparation_seconds": time.monotonic() - started,
        "competition_submission": False,
    }
    if not secondary and total == 1537638:
        assert train.height == 1230110 and valid.height == 307528
    y_train = torch.tensor(train[TARGET].to_numpy().copy(), dtype=torch.float32)
    y_valid = valid[TARGET].to_numpy().copy()
    del sample, train, valid, identity
    gc.collect()
    joblib.dump(state, output / "preprocessing.joblib")
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    return {
        "x_train": x_train, "c_train": c_train, "x_valid": x_valid, "c_valid": c_valid,
        "y_train": y_train, "y_valid": y_valid, "state": state, "protocol": protocol,
        "original_train": original_train, "original_valid": original_valid,
        "category_counts": category_counts,
    }


def category_view(data, minimum=10, drop_high=False):
    state = data["state"]
    if minimum == 10 and not drop_high:
        return data["c_train"], data["c_valid"], state["cardinalities"]
    train = np.empty(data["c_train"].shape, dtype=np.int64)
    valid = np.empty(data["c_valid"].shape, dtype=np.int64)
    cardinalities = []
    for index, counts in enumerate(data["category_counts"]):
        retained = np.flatnonzero((counts >= minimum) & (np.arange(len(counts)) != 0))
        if drop_high and state["original_cardinalities"][index] > 1000:
            retained = np.empty(0, dtype=np.int64)
        mapping = np.zeros(len(counts), dtype=np.int64)
        mapping[retained] = np.arange(1, len(retained) + 1)
        train[:, index] = mapping[data["original_train"][index]]
        valid[:, index] = mapping[data["original_valid"][index]]
        cardinalities.append(len(retained) + 1)
    return torch.from_numpy(train), torch.from_numpy(valid), cardinalities


def run_tabm(data, spec, output, *, device="cuda", epochs=30, interval=300, seed=42):
    name = spec["name"]
    destination = output / name
    destination.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    train_categories, valid_categories, cardinalities = category_view(
        data, spec.get("minimum", 10), spec.get("drop_high", False)
    )
    dims = [
        min(32, math.ceil(cardinality ** 0.25)) if spec.get("adaptive") else spec.get("embedding_dim", 8)
        for cardinality in cardinalities
    ]
    config = {
        "n_numeric": data["x_train"].shape[1], "cardinalities": cardinalities,
        "members": spec.get("members", 8), "width": spec.get("width", 128),
        "blocks": 2, "dropout": spec.get("dropout", 0.2),
        "category_dropout": 0.3, "positive_rate": float(data["y_train"].mean().item()),
        "embedding_dimensions": dims,
        "numeric_dropout": spec.get("numeric_dropout", 0.0),
        "dropout_columns": data["protocol"]["continuous_indices"],
    }
    model = TabMClassifier(**config).to(device)
    learning_rate = spec.get("learning_rate", 0.0005)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=spec.get("weight_decay", 0.01)
    )
    scheduler = None
    if spec.get("cosine"):
        period = 2 * math.ceil(len(data["y_train"]) / 1024)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: 0.1 + 0.45 * (1 + math.cos(math.pi * min(step / period, 1)))
        )
    print(f"START {name}: {json.dumps(spec)}", flush=True)
    metrics = fit_model_steps(
        model, config, optimizer, data["x_train"], train_categories, data["y_train"],
        data["x_valid"], valid_categories, data["y_valid"],
        device=device, output=destination, epochs=epochs, patience=5, batch_size=1024,
        validation_interval=interval, seed=seed, scheduler=scheduler,
    )
    result = {
        "name": name, "spec": spec, "seed": seed,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "model_config": config, **metrics,
    }
    (destination / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"RESULT {name}: AUC={result['validation_auc']:.10f}", flush=True)
    del model, optimizer, train_categories, valid_categories
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return result


def optimization_specs():
    return [
        {"name": "epoch-checkpoint-control", "epoch_only": True},
        {"name": "reference"},
        {"name": "learning-rate-1e4", "learning_rate": 0.0001},
        {"name": "dropout-035", "dropout": 0.35},
        {"name": "weight-decay-003", "weight_decay": 0.03},
        {"name": "width-64", "width": 64},
        {"name": "numeric-dropout-01", "numeric_dropout": 0.1},
        {"name": "cosine-two-epochs", "cosine": True},
        {"name": "minimum-category-50", "minimum": 50},
        {"name": "embedding-dim-4", "embedding_dim": 4},
        {"name": "adaptive-category-dims", "adaptive": True},
        {"name": "without-high-cardinality", "drop_high": True},
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--stage", choices=["optimization"], default="optimization")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--validation-interval", type=int, default=300)
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The full Kaggle experiment requires CUDA")
    torch.set_num_threads(4)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = prepare_data(args.data_dir / "train.parquet", args.output_dir)
    report = {"stage": args.stage, "protocol": data["protocol"], "results": [], "competition_submission": False}
    for spec in optimization_specs():
        result = run_tabm(
            data, spec, args.output_dir, device=args.device,
            epochs=args.epochs,
            interval=math.ceil(len(data["y_train"]) / 1024) if spec.get("epoch_only") else args.validation_interval,
        )
        report["results"].append(result)
        (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    report["status"] = "COMPLETE"
    (args.output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
