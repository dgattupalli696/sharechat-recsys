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
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import torch
from torch import nn
from torch.nn import functional as F

from train import ROOT, TARGET, load_sample


def prepare_features(train, valid):
    columns = [name for name in train.columns if name not in {TARGET, "_time"}]
    numeric = [name for name in columns if train.schema[name].is_numeric()]
    categorical = [name for name in columns if train.schema[name] == pl.String]
    if len(numeric) + len(categorical) != len(columns):
        raise TypeError("The engineered sample must contain only numeric/string features")
    imputer = SimpleImputer(
        strategy="median", keep_empty_features=True, add_indicator=True, copy=False
    )
    scaler = StandardScaler(copy=False)
    transformed, infinity_counts = [], []
    for label, frame in [("training", train), ("validation", valid)]:
        array = np.array(frame.select(numeric).to_numpy(), dtype=np.float32, order="C")
        count = int(np.isinf(array).sum())
        infinity_counts.append(count)
        if count:
            print(f"Treating {count} infinite {label} values as missing", flush=True)
            array[np.isinf(array)] = np.nan
        if label == "training":
            array = imputer.fit_transform(array)
            array = scaler.fit_transform(array)
        else:
            array = imputer.transform(array)
            array = scaler.transform(array)
        transformed.append(array)
    x_train, x_valid = transformed
    for array in [x_train, x_valid]:
        np.clip(array, -10, 10, out=array)
        if not np.isfinite(array).all():
            raise ValueError("Nonfinite numerical features after preprocessing")
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64
    )
    categories = [
        frame.select(categorical).fill_null("__MISSING__").to_numpy()
        for frame in [train, valid]
    ]
    c_train = encoder.fit_transform(categories[0]) + 1
    c_valid = encoder.transform(categories[1]) + 1
    cardinalities = [len(values) + 1 for values in encoder.categories_]
    preprocessing = {
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "imputer": imputer,
        "scaler": scaler,
        "encoder": encoder,
        "cardinalities": cardinalities,
        "infinity_counts": infinity_counts,
        "validation_unknown_category_rates": {
            name: float((c_valid[:, index] == 0).mean())
            for index, name in enumerate(categorical)
        },
    }
    return (
        torch.from_numpy(np.ascontiguousarray(x_train, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(c_train)),
        torch.from_numpy(np.ascontiguousarray(x_valid, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(c_valid)),
        preprocessing,
    )


def transform_features(frame, state):
    numeric = np.array(
        frame.select(state["numeric_columns"]).to_numpy(),
        dtype=np.float32,
        order="C",
    )
    infinities = np.isinf(numeric)
    if infinities.any():
        print(f"Treating {int(infinities.sum())} infinite inference values as missing", flush=True)
        numeric[infinities] = np.nan
    numeric = state["scaler"].transform(state["imputer"].transform(numeric))
    np.clip(numeric, -10, 10, out=numeric)
    if not np.isfinite(numeric).all():
        raise ValueError("Nonfinite numerical features after inference preprocessing")
    categories = (
        frame.select(state["categorical_columns"]).fill_null("__MISSING__").to_numpy()
    )
    encoded = state["encoder"].transform(categories) + 1
    return (
        torch.from_numpy(np.ascontiguousarray(numeric, dtype=np.float32)),
        torch.from_numpy(np.ascontiguousarray(encoded, dtype=np.int64)),
    )


class TabMClassifier(nn.Module):
    def __init__(
        self, n_numeric, cardinalities, members=8, width=128, blocks=2, dropout=0.1,
        category_dropout=0.0, positive_rate=None,
        embedding_dimensions=None, numeric_dropout=0.0, dropout_columns=None,
    ):
        super().__init__()
        from tabm import TabM

        if not 0 <= category_dropout <= 1:
            raise ValueError("category_dropout must be between zero and one")
        self.category_dropout = category_dropout
        self.numeric_dropout = numeric_dropout
        self.dropout_columns = dropout_columns or []
        dimensions = embedding_dimensions or [8] * len(cardinalities)
        if len(dimensions) != len(cardinalities) or any(d <= 0 for d in dimensions):
            raise ValueError("Expected one positive embedding dimension per category")
        self.embeddings = nn.ModuleList(
            nn.Embedding(cardinality, dimension, padding_idx=0)
            for cardinality, dimension in zip(cardinalities, dimensions)
        )
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, std=0.02)
            with torch.no_grad():
                embedding.weight[0].zero_()
        self.backbone = TabM.make(
            n_num_features=n_numeric + sum(dimensions),
            d_out=1,
            k=members,
            n_blocks=blocks,
            d_block=width,
            dropout=dropout,
        )
        if positive_rate is not None:
            if not 0 < positive_rate < 1:
                raise ValueError("positive_rate must be strictly between zero and one")
            with torch.no_grad():
                self.backbone.output.bias.fill_(
                    float(np.log(positive_rate / (1 - positive_rate)))
                )

    def forward(self, numeric, categorical):
        if self.training and self.numeric_dropout:
            numeric = numeric.clone()
            numeric[:, self.dropout_columns] = F.dropout(
                numeric[:, self.dropout_columns], p=self.numeric_dropout, training=True
            )
        if self.training and self.category_dropout:
            categorical = categorical.masked_fill(
                torch.rand_like(categorical, dtype=torch.float32) < self.category_dropout,
                0,
            )
        embedded = [
            layer(categorical[:, index])
            for index, layer in enumerate(self.embeddings)
        ]
        return self.backbone(torch.cat([numeric, *embedded], dim=1)).squeeze(-1)


@torch.inference_mode()
def predict(model, numeric, categorical, device, batch_size):
    model.eval()
    probabilities = []
    for start in range(0, len(numeric), batch_size):
        logits = model(
            numeric[start : start + batch_size].to(device),
            categorical[start : start + batch_size].to(device),
        )
        probabilities.append(logits.sigmoid().mean(dim=1).cpu().numpy())
    return np.concatenate(probabilities)


def fit_model(
    model, model_config, optimizer, x_train, c_train, y_train, x_valid, c_valid, y_valid,
    *, device, output, epochs, patience, batch_size,
):
    generator = torch.Generator().manual_seed(42)
    best_auc, best_epoch, stale = -np.inf, 0, 0
    history = []
    training_started = time.monotonic()
    for epoch in range(1, epochs + 1):
        epoch_started = time.monotonic()
        model.train()
        order = torch.randperm(len(y_train), generator=generator)
        loss_sum = 0.0
        for indices in order.split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_train[indices].to(device), c_train[indices].to(device))
            targets = y_train[indices].to(device).unsqueeze(1).expand_as(logits)
            loss = F.binary_cross_entropy_with_logits(logits, targets)
            if not torch.isfinite(loss).item():
                raise ValueError(f"Nonfinite training loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(indices)
        probabilities = predict(model, x_valid, c_valid, device, batch_size)
        auc = float(roc_auc_score(y_valid, probabilities))
        record = {
            "epoch": epoch,
            "training_loss": loss_sum / len(y_train),
            "validation_auc": auc,
            "seconds": time.monotonic() - epoch_started,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        if auc > best_auc:
            best_auc, best_epoch, stale = auc, epoch, 0
            torch.save(
                {
                    "model_config": model_config,
                    "state_dict": {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    },
                },
                output / "best_model.pt",
            )
        else:
            stale += 1
            if stale >= patience:
                break
    training_seconds = time.monotonic() - training_started
    checkpoint = torch.load(output / "best_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    probabilities = predict(model, x_valid, c_valid, device, batch_size)
    restored_auc = float(roc_auc_score(y_valid, probabilities))
    if abs(restored_auc - best_auc) > 1e-5:
        raise ValueError("Reloaded checkpoint does not reproduce the selected validation AUC")
    np.savez_compressed(
        output / "validation_predictions.npz", y_true=y_valid, probabilities=probabilities
    )
    return {
        "validation_auc": restored_auc,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "training_seconds": training_seconds,
        "history": history,
    }


def fit_model_steps(
    model, model_config, optimizer, x_train, c_train, y_train, x_valid, c_valid, y_valid,
    *, device, output, epochs=30, patience=5, batch_size=1024, validation_interval=300,
    seed=42, scheduler=None,
):
    if min(epochs, patience, batch_size, validation_interval) <= 0:
        raise ValueError("Training budgets and validation interval must be positive")
    output.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)
    probe_indices = torch.randperm(
        len(y_train), generator=torch.Generator().manual_seed(42)
    )[: min(16384, len(y_train))]
    steps_per_epoch = (len(y_train) + batch_size - 1) // batch_size
    best_auc, best_step, global_step = -np.inf, 0, 0
    history, checks = [], []
    started = time.monotonic()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, rows_seen, stop = 0.0, 0, False
        order = torch.randperm(len(y_train), generator=generator)
        for batch_index, indices in enumerate(order.split(batch_size), 1):
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_train[indices].to(device), c_train[indices].to(device))
            targets = y_train[indices].to(device).unsqueeze(1).expand_as(logits)
            loss = F.binary_cross_entropy_with_logits(logits, targets)
            if not torch.isfinite(loss).item():
                raise ValueError(f"Nonfinite loss at epoch {epoch}, step {global_step}")
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            global_step += 1
            loss_sum += loss.item() * len(indices)
            rows_seen += len(indices)
            expired = global_step - best_step >= patience * steps_per_epoch
            if (
                global_step % validation_interval != 0
                and batch_index != steps_per_epoch and not expired
            ):
                continue
            probabilities = predict(model, x_valid, c_valid, device, batch_size)
            auc = float(roc_auc_score(y_valid, probabilities))
            probe = predict(
                model, x_train[probe_indices], c_train[probe_indices], device, batch_size
            )
            probe_y = y_train[probe_indices].numpy()
            record = {
                "epoch": epoch,
                "global_step": global_step,
                "fractional_epoch": epoch - 1 + batch_index / steps_per_epoch,
                "validation_auc": auc,
                "validation_logloss": float(log_loss(y_valid, probabilities, labels=[0, 1])),
                "training_probe_auc": float(roc_auc_score(probe_y, probe)),
                "training_probe_logloss": float(log_loss(probe_y, probe, labels=[0, 1])),
                "learning_rate": optimizer.param_groups[0]["lr"],
                "seconds": time.monotonic() - started,
            }
            checks.append(record)
            print(json.dumps(record), flush=True)
            if auc > best_auc:
                best_auc, best_step = auc, global_step
                torch.save(
                    {
                        "model_config": model_config,
                        "state_dict": {
                            key: value.detach().cpu().clone()
                            for key, value in model.state_dict().items()
                        },
                        "selection": record,
                    },
                    output / "best_model.pt",
                )
            model.train()
            if global_step - best_step >= patience * steps_per_epoch:
                stop = True
                break
        history.append({
            "epoch": epoch, "training_loss": loss_sum / rows_seen,
            "last_validation_auc": checks[-1]["validation_auc"],
        })
        if stop:
            break
    checkpoint = torch.load(output / "best_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    probabilities = predict(model, x_valid, c_valid, device, batch_size)
    auc = float(roc_auc_score(y_valid, probabilities))
    if abs(auc - best_auc) > 1e-5:
        raise ValueError("Fine checkpoint reload does not reproduce its validation AUC")
    np.savez_compressed(
        output / "validation_predictions.npz", y_true=y_valid, probabilities=probabilities
    )
    return {
        "validation_auc": auc,
        "validation_logloss": float(log_loss(y_valid, probabilities, labels=[0, 1])),
        "best_epoch": checkpoint["selection"]["epoch"],
        "best_global_step": best_step,
        "best_fractional_epoch": checkpoint["selection"]["fractional_epoch"],
        "optimizer_steps": global_step, "epochs_run": len(history),
        "training_seconds": time.monotonic() - started,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
        "history": history, "checkpoint_history": checks,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=150_000)
    parser.add_argument("--sample-path", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--members", type=int, default=8)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/local-tabm-v1")
    args = parser.parse_args()
    if min(
        args.rows, args.epochs, args.patience, args.batch_size,
        args.members, args.width, args.blocks, args.threads,
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
    torch.manual_seed(42)
    started = time.monotonic()
    sample_hash = None
    if args.sample_path:
        sample = pl.read_parquet(args.sample_path)
        if sample.height != args.rows:
            raise ValueError("Cached sample row count does not match --rows")
        with args.sample_path.open("rb") as stream:
            sample_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    else:
        sample = load_sample(ROOT / "data/train.parquet", args.rows, 42)
    sample = sample.select(
        name for name in sample.columns if "_dim_" not in name
    ).sort("_time")
    if sample.height < 2 or sample["_time"].null_count():
        raise ValueError("Chronological validation requires nonmissing timestamps")
    cutoff = sample["_time"][int(sample.height * 0.8)]
    train = sample.filter(pl.col("_time") < cutoff)
    valid = sample.filter(pl.col("_time") >= cutoff)
    for label, frame in [("training", train), ("validation", valid)]:
        if set(frame[TARGET].unique().to_list()) != {0, 1}:
            raise ValueError(f"{label} must contain both binary target classes")
    training_rows, validation_rows = train.height, valid.height
    y_train = torch.tensor(train[TARGET].to_numpy().copy(), dtype=torch.float32)
    y_valid = valid[TARGET].to_numpy().copy()
    x_train, c_train, x_valid, c_valid, preprocessing = prepare_features(train, valid)
    feature_count = sample.width - 2
    sample_rows = sample.height
    del sample, train, valid
    gc.collect()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessing, output / "preprocessing.joblib")
    model_config = {
        "n_numeric": x_train.shape[1],
        "cardinalities": preprocessing["cardinalities"],
        "members": args.members,
        "width": args.width,
        "blocks": args.blocks,
        "dropout": 0.1,
    }
    model = TabMClassifier(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.0003)
    preparation_seconds = time.monotonic() - started
    print(
        f"Device: {device}; sample: {sample_rows:,}; train: {training_rows:,}; "
        f"valid: {validation_rows:,}; features: {feature_count}; "
        f"cutoff: {cutoff}; preparation: {preparation_seconds:.1f}s",
        flush=True,
    )
    fitted = fit_model(
        model, model_config, optimizer, x_train, c_train, y_train, x_valid, c_valid, y_valid,
        device=device, output=output, epochs=args.epochs, patience=args.patience,
        batch_size=args.batch_size,
    )
    report = {
        "algorithm": "TabM with learned categorical embeddings",
        **fitted,
        "seed": 42,
        "sample_rows": sample_rows,
        "sample_sha256": sample_hash,
        "training_rows": training_rows,
        "validation_rows": validation_rows,
        "cutoff": str(cutoff),
        "input_features": feature_count,
        "numerical_features": len(preprocessing["numeric_columns"]),
        "numerical_features_after_imputation": x_train.shape[1],
        "categorical_features": len(preprocessing["categorical_columns"]),
        "validation_unknown_category_rates": preprocessing["validation_unknown_category_rates"],
        "infinite_input_counts_train_valid": preprocessing["infinity_counts"],
        "model_config": model_config,
        "optimizer": {"name": "AdamW", "learning_rate": 0.002, "weight_decay": 0.0003},
        "max_epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "device": device,
        "threads": args.threads,
        "preparation_seconds": preparation_seconds,
        "versions": {
            name: version(name)
            for name in ["torch", "tabm", "numpy", "polars", "scikit-learn"]
        },
        "competition_submission": False,
    }
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Best validation ROC AUC: {fitted['validation_auc']:.6f}; "
        f"epoch: {fitted['best_epoch']}; training: {fitted['training_seconds']:.1f}s; "
        "no submission performed",
        flush=True,
    )


if __name__ == "__main__":
    main()
