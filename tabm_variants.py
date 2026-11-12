import argparse
import gc
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

import joblib
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch

from custom_network import RecommendationNetwork, prepare_custom_features
from tabm_experiment import TabMClassifier, fit_model
from train import ROOT, TARGET, load_sample


class AttentionTabMClassifier(RecommendationNetwork):
    def __init__(
        self, n_dense, cardinalities, n_vectors, positive_rate=0.05,
        members=8, width=128, blocks=2, dropout=0.2,
    ):
        super().__init__(
            n_dense, cardinalities, n_vectors,
            attention=True, positive_rate=positive_rate,
        )
        from tabm import TabM

        self.head = TabM.make(
            n_num_features=256, d_out=1, k=members,
            n_blocks=blocks, d_block=width, dropout=dropout,
        )
        with torch.no_grad():
            self.head.output.bias.fill_(
                float(np.log(positive_rate / (1 - positive_rate)))
            )

    def forward(self, numeric, categorical):
        return self.head(self.extract_features(numeric, categorical)).squeeze(-1)


def main():
    parser = argparse.ArgumentParser(
        description="Validate updated TabM variants on a fixed chronological holdout."
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--sample-path", type=Path, help="Pre-engineered dataset for replay")
    parser.add_argument("--training-rows", type=int, help="Sample training only; keep all holdout rows")
    parser.add_argument("--variant", choices=["both", "standalone", "hybrid"], default="both")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--members", type=int, default=8)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--min-category-count", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/tabm-variants-v1")
    args = parser.parse_args()
    if min(
        args.epochs, args.patience, args.batch_size, args.members,
        args.width, args.blocks, args.min_category_count, args.threads,
    ) <= 0 or (args.training_rows is not None and args.training_rows <= 0):
        parser.error("All numerical arguments must be positive")
    device = args.device
    if device == "auto":
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS requested but unavailable")
    torch.set_num_threads(args.threads)
    started = time.monotonic()
    sample_hash = None
    if args.sample_path:
        sample = pl.read_parquet(args.sample_path)
        with args.sample_path.open("rb") as stream:
            sample_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    else:
        path = args.data_dir / "train.parquet"
        rows = pq.ParquetFile(path).metadata.num_rows
        sample = load_sample(path, rows, 42, expand_embeddings=True)
    sample = sample.sort("_time")
    if sample.height < 2 or sample["_time"].null_count():
        raise ValueError("Chronological validation requires nonmissing timestamps")
    cutoff = sample["_time"][int(sample.height * 0.8)]
    split = int(sample["_time"].search_sorted(cutoff, side="left"))
    train, valid = sample.slice(0, split), sample.slice(split)
    available_training_rows = train.height
    if args.training_rows is not None:
        if args.training_rows > available_training_rows:
            raise ValueError("--training-rows exceeds the available training partition")
        train = train.sample(n=args.training_rows, seed=42).sort("_time")
    for label, frame in [("training", train), ("validation", valid)]:
        if set(frame[TARGET].unique().to_list()) != {0, 1}:
            raise ValueError(f"{label} must contain both binary target classes")
    x_train, c_train, x_valid, c_valid, state = prepare_custom_features(
        train, valid, args.min_category_count
    )
    y_train = torch.tensor(train[TARGET].to_numpy().copy(), dtype=torch.float32)
    y_valid = valid[TARGET].to_numpy().copy()
    report = {
        "seed": 42,
        "sample_rows": sample.height,
        "sample_sha256": sample_hash,
        "available_training_rows": available_training_rows,
        "training_rows": train.height,
        "validation_rows": valid.height,
        "cutoff": str(cutoff),
        "validation_labels_sha256": hashlib.sha256(
            y_valid.astype(np.uint8).tobytes()
        ).hexdigest(),
        "validation_timestamps_sha256": hashlib.sha256(
            valid["_time"].cast(pl.Int64).to_numpy().tobytes()
        ).hexdigest(),
        "input_features": sample.width - 2,
        "dense_channels_after_preprocessing": state["n_dense"],
        "numeric_channels_with_vectors": x_train.shape[1],
        "vector_fields": state["vector_fields"],
        "original_cardinalities": state["original_cardinalities"],
        "retained_cardinalities": state["cardinalities"],
        "min_category_count": args.min_category_count,
        "category_dropout": 0.3,
        "validation_unknown_or_rare_category_rates": state["validation_unknown_or_rare_category_rates"],
        "missing_vectors_train": state["missing_vectors_train"],
        "missing_vectors_valid": state["missing_vectors_valid"],
        "optimizer": {"name": "AdamW", "learning_rate": 0.0005, "weight_decay": 0.01},
        "max_epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "device": device,
        "threads": args.threads,
        "versions": {
            name: version(name)
            for name in ["torch", "tabm", "rtdl_num_embeddings", "numpy", "polars", "scikit-learn"]
        },
        "competition_submission": False,
        "results": [],
    }
    positive_rate = float(y_train.mean().item())
    del sample, train, valid
    gc.collect()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(state, output / "preprocessing.joblib")
    report["preparation_seconds"] = time.monotonic() - started
    print(
        f"Prepared {len(y_train):,} training / {len(y_valid):,} validation rows; "
        f"cutoff: {cutoff}; preparation: {report['preparation_seconds']:.1f}s",
        flush=True,
    )
    variants = ["standalone", "hybrid"] if args.variant == "both" else [args.variant]
    for variant in variants:
        torch.manual_seed(42)
        model_config = {
            "cardinalities": state["cardinalities"],
            "positive_rate": positive_rate,
            "members": args.members,
            "width": args.width,
            "blocks": args.blocks,
            "dropout": 0.2,
        }
        if variant == "standalone":
            model_config.update(n_numeric=x_train.shape[1], category_dropout=0.3)
            model = TabMClassifier(**model_config).to(device)
        else:
            model_config.update(n_dense=state["n_dense"], n_vectors=len(state["vector_fields"]))
            model = AttentionTabMClassifier(**model_config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
        variant_output = output / variant
        variant_output.mkdir(exist_ok=True)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        print(f"Variant: {variant}; device: {device}; parameters: {parameters:,}", flush=True)
        fitted = fit_model(
            model, model_config, optimizer, x_train, c_train, y_train,
            x_valid, c_valid, y_valid, device=device, output=variant_output,
            epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
        )
        report["results"].append(
            {"variant": variant, "parameters": parameters, "model_config": model_config, **fitted}
        )
        (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"{variant}: AUC={fitted['validation_auc']:.6f}; "
            f"epoch={fitted['best_epoch']}; training={fitted['training_seconds']:.1f}s",
            flush=True,
        )
        del model, optimizer
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()
    print("Requested TabM validation variants saved; no competition submission performed", flush=True)


if __name__ == "__main__":
    main()
