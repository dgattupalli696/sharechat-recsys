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
import torch
from torch import nn
from torch.nn import functional as F

from tabm_experiment import fit_model, prepare_features
from train import ROOT, TARGET, load_sample


def pack_vectors(frame, fields):
    packed, missing = [], {}
    for field in fields:
        columns = [f"{field}_dim_{index}" for index in range(32)]
        raw = np.array(frame.select(columns).to_numpy(), dtype=np.float32, order="C")
        valid = np.isfinite(raw).all(axis=1)
        missing[field] = int((~valid).sum())
        raw[~valid] = 0
        norm = np.linalg.norm(raw.astype(np.float64), axis=1)
        unit = (raw / np.maximum(norm, 1e-8)[:, None]).astype(np.float32)
        log_norm = np.clip(np.log1p(norm), 0, 20).astype(np.float32)
        packed.append(
            np.concatenate([unit, log_norm[:, None], valid[:, None]], axis=1)
        )
    return torch.from_numpy(np.stack(packed, axis=1).astype(np.float32)), missing


def prepare_custom_features(train, valid, min_category_count):
    dimension_columns = [name for name in train.columns if "_dim_" in name]
    fields = sorted(
        {name.rsplit("_dim_", 1)[0] for name in dimension_columns},
        key=lambda name: int(name.removeprefix("feature")),
    )
    if not fields or set(dimension_columns) != {
        f"{field}_dim_{index}" for field in fields for index in range(32)
    }:
        raise ValueError("Expected complete 32-dimensional vector fields")
    summary = [name for name in train.columns if "_dim_" not in name]
    x_train, c_train, x_valid, c_valid, state = prepare_features(
        train.select(summary), valid.select(summary)
    )
    mappings, cardinalities = [], []
    mapped_train = np.empty(c_train.shape, dtype=np.int64)
    mapped_valid = np.empty(c_valid.shape, dtype=np.int64)
    for index, cardinality in enumerate(state["cardinalities"]):
        counts = np.bincount(c_train[:, index].numpy(), minlength=cardinality)
        retained = np.flatnonzero(
            (counts >= min_category_count) & (np.arange(cardinality) != 0)
        )
        mapping = np.zeros(cardinality, dtype=np.int64)
        mapping[retained] = np.arange(1, len(retained) + 1)
        mapped_train[:, index] = mapping[c_train[:, index].numpy()]
        mapped_valid[:, index] = mapping[c_valid[:, index].numpy()]
        mappings.append(mapping)
        cardinalities.append(len(retained) + 1)
    train_vectors, missing_train = pack_vectors(train, fields)
    valid_vectors, missing_valid = pack_vectors(valid, fields)
    n_dense = x_train.shape[1]
    x_train = torch.cat([x_train, train_vectors.flatten(1)], dim=1)
    x_valid = torch.cat([x_valid, valid_vectors.flatten(1)], dim=1)
    state.update(
        {
            "original_cardinalities": state["cardinalities"],
            "cardinalities": cardinalities,
            "category_remappings": mappings,
            "min_category_count": min_category_count,
            "vector_fields": fields,
            "n_dense": n_dense,
            "missing_vectors_train": missing_train,
            "missing_vectors_valid": missing_valid,
            "validation_unknown_or_rare_category_rates": {
                name: float((mapped_valid[:, index] == 0).mean())
                for index, name in enumerate(state["categorical_columns"])
            },
        }
    )
    return (
        x_train, torch.from_numpy(mapped_train),
        x_valid, torch.from_numpy(mapped_valid), state,
    )


class RecommendationNetwork(nn.Module):
    def __init__(
        self, n_dense, cardinalities, n_vectors, attention=False, positive_rate=0.05
    ):
        super().__init__()
        self.n_dense = n_dense
        self.n_vectors = n_vectors
        self.embeddings = nn.ModuleList(
            nn.Embedding(cardinality, 8, padding_idx=0)
            for cardinality in cardinalities
        )
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, std=0.02)
            with torch.no_grad():
                embedding.weight[0].zero_()
        self.scalar_branch = nn.Sequential(
            nn.Linear(n_dense, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2)
        )
        self.category_branch = nn.Sequential(
            nn.Linear(8 * len(cardinalities), 64),
            nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.3),
        )
        self.vector_weights = nn.Parameter(torch.empty(n_vectors, 33, 32))
        for weight in self.vector_weights:
            nn.init.xavier_uniform_(weight)
        self.vector_bias = nn.Parameter(torch.zeros(n_vectors, 32))
        self.missing_tokens = nn.Parameter(torch.randn(n_vectors, 32) * 0.02)
        self.field_tokens = nn.Parameter(torch.randn(n_vectors, 32) * 0.02)
        self.vector_transform = nn.Sequential(
            nn.LayerNorm(32), nn.Linear(32, 64), nn.GELU(),
            nn.Dropout(0.2), nn.Linear(64, 32),
        )
        self.vector_branch = nn.Sequential(
            nn.Linear(n_vectors * 32, 64),
            nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.2),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(0.3), nn.Linear(64, 1)
        )
        with torch.no_grad():
            self.head[-1].bias.fill_(float(np.log(positive_rate / (1 - positive_rate))))
        self.attention = (
            nn.MultiheadAttention(32, 4, dropout=0.2, batch_first=True)
            if attention else None
        )
        self.attention_norm = nn.LayerNorm(32) if attention else None
        self.attention_dropout = nn.Dropout(0.2)

    def forward(self, numeric, categorical):
        if self.training:
            categorical = categorical.masked_fill(
                torch.rand_like(categorical, dtype=torch.float32) < 0.3, 0
            )
        embedded = torch.cat(
            [
                layer(categorical[:, index])
                for index, layer in enumerate(self.embeddings)
            ],
            dim=1,
        )
        vectors = numeric[:, self.n_dense :].reshape(-1, self.n_vectors, 34)
        projected = F.gelu(
            torch.einsum("bfi,fij->bfj", vectors[:, :, :33], self.vector_weights)
            + self.vector_bias
        )
        tokens = torch.where(
            vectors[:, :, 33:].bool(), projected, self.missing_tokens.unsqueeze(0)
        ) + self.field_tokens
        if self.attention is not None:
            normalized = self.attention_norm(tokens)
            interactions, _ = self.attention(
                normalized, normalized, normalized, need_weights=False
            )
            tokens = tokens + self.attention_dropout(interactions)
        tokens = tokens + self.vector_transform(tokens)
        fused = torch.cat(
            [
                self.scalar_branch(numeric[:, : self.n_dense]),
                self.category_branch(embedded),
                self.vector_branch(tokens.flatten(1)),
            ],
            dim=1,
        )
        return self.head(fused)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=150_000)
    parser.add_argument("--sample-path", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--min-category-count", type=int, default=10)
    parser.add_argument("--variant", choices=["both", "control", "attention"], default="both")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/local-custom-v1")
    args = parser.parse_args()
    if min(
        args.rows, args.epochs, args.patience, args.batch_size,
        args.min_category_count, args.threads,
    ) <= 0:
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
        if sample.height != args.rows:
            raise ValueError("Cached sample row count does not match --rows")
        with args.sample_path.open("rb") as stream:
            sample_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    else:
        sample = load_sample(
            ROOT / "data/train.parquet", args.rows, 42, expand_embeddings=True
        )
    sample = sample.sort("_time")
    if sample.height < 2 or sample["_time"].null_count():
        raise ValueError("Chronological validation requires nonmissing timestamps")
    cutoff = sample["_time"][int(sample.height * 0.8)]
    train = sample.filter(pl.col("_time") < cutoff)
    valid = sample.filter(pl.col("_time") >= cutoff)
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
        "training_rows": train.height,
        "validation_rows": valid.height,
        "cutoff": str(cutoff),
        "input_features": sample.width - 2,
        "dense_channels_after_preprocessing": state["n_dense"],
        "vector_fields": state["vector_fields"],
        "original_cardinalities": state["original_cardinalities"],
        "retained_cardinalities": state["cardinalities"],
        "min_category_count": args.min_category_count,
        "validation_unknown_or_rare_category_rates": state["validation_unknown_or_rare_category_rates"],
        "missing_vectors_train": state["missing_vectors_train"],
        "missing_vectors_valid": state["missing_vectors_valid"],
        "device": device,
        "threads": args.threads,
        "max_epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "optimizer": {"name": "AdamW", "learning_rate": 0.0005, "weight_decay": 0.01},
        "versions": {
            name: version(name)
            for name in ["torch", "numpy", "polars", "scikit-learn"]
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
    variants = ["control", "attention"] if args.variant == "both" else [args.variant]
    for variant in variants:
        torch.manual_seed(42)
        model_config = {
            "n_dense": state["n_dense"],
            "cardinalities": state["cardinalities"],
            "n_vectors": len(state["vector_fields"]),
            "attention": variant == "attention",
            "positive_rate": positive_rate,
        }
        model = RecommendationNetwork(**model_config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
        variant_output = output / variant
        variant_output.mkdir(exist_ok=True)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        print(
            f"Variant: {variant}; device: {device}; train: {len(y_train):,}; "
            f"valid: {len(y_valid):,}; parameters: {parameters:,}; cutoff: {cutoff}",
            flush=True,
        )
        fitted = fit_model(
            model, model_config, optimizer, x_train, c_train, y_train,
            x_valid, c_valid, y_valid, device=device, output=variant_output,
            epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
        )
        result = {
            "variant": variant,
            "parameters": parameters,
            "model_config": model_config,
            **fitted,
        }
        report["results"].append(result)
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
    print("Both results saved; no Kaggle run or competition submission performed", flush=True)


if __name__ == "__main__":
    main()
