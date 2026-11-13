"""Feed-attention model: rows of one request attend to each other before a TabM head.

Each row is encoded with FeatureEncoder; rows sharing a request_id (one feed, at most
~24 videos) exchange information through masked self-attention, and explicit
leave-one-out cosine similarities between each row's embedding vectors and the rest of
its feed are added. The TabM head sees [row encoding, attended context, similarities],
so it is a strict superset of the T024 k32 model.
"""
import argparse
from datetime import datetime
import gc
import hashlib
import json
import math
from pathlib import Path
import time

import joblib
import numpy as np
import polars as pl
import pyarrow.parquet as pq
from sklearn.metrics import log_loss, roc_auc_score
import torch
from torch import nn
from torch.nn import functional as F

from feature_models import FeatureEncoder

CUTOFF = "2026-10-28T08:06:52.397000"


def feed_similarity(numeric, groups, n_dense, n_vectors):
    """Leave-one-out cosine of each row's vectors with the rest of its feed."""
    vectors = numeric[:, n_dense:].reshape(-1, n_vectors, 34)
    present = vectors[:, :, 33:34]
    unit = vectors[:, :, :32] * present
    count = int(groups.max().item()) + 1
    totals = unit.new_zeros(count, n_vectors, 32).index_add_(0, groups, unit)
    present_totals = present.new_zeros(count, n_vectors, 1).index_add_(0, groups, present)
    others = totals[groups] - unit
    others_present = (present_totals[groups] - present).squeeze(-1)
    norm = others.norm(dim=-1)
    cosine = (unit * others).sum(dim=-1) / norm.clamp_min(1e-6)
    coherence = norm / others_present.clamp_min(1)
    has = present.squeeze(-1) * (others_present > 0)
    return torch.cat([cosine * has, coherence * has, others_present.log1p()], dim=1)


class FeedAttentionClassifier(nn.Module):
    def __init__(
        self, encoder_config, width=128, dropout=0.2, members=32, positive_rate=0.05,
        heads=4, layers=2, attention_dropout=0.1, similarity=True,
    ):
        super().__init__()
        from tabm import TabM

        self.encoder = FeatureEncoder(**encoder_config)
        self.n_dense, self.n_vectors = encoder_config["n_dense"], encoder_config["n_vectors"]
        self.similarity = similarity
        dimension = self.encoder.output_dim
        self.project = nn.Sequential(
            nn.Linear(dimension, width), nn.ReLU(), nn.Dropout(dropout), nn.Linear(width, width),
        )
        self.attention = nn.ModuleList(
            nn.TransformerEncoderLayer(
                width, heads, 2 * width, attention_dropout, batch_first=True, norm_first=True,
            ) for _ in range(layers)
        )
        self.norm = nn.LayerNorm(width)
        extra = 3 * self.n_vectors if similarity else 0
        self.backbone = TabM.make(
            n_num_features=dimension + width + extra, d_out=1, k=members,
            n_blocks=2, d_block=width, dropout=dropout,
        )
        with torch.no_grad():
            self.backbone.output.bias.fill_(float(np.log(positive_rate / (1 - positive_rate))))

    def forward(self, numeric, categorical, groups):
        """groups: local request index per row; rows of one request must be contiguous."""
        encoded = self.encoder(numeric, categorical)
        hidden = self.project(encoded)
        count = int(groups[-1].item()) + 1
        sizes = torch.bincount(groups, minlength=count)
        starts = torch.cumsum(sizes, 0) - sizes
        position = torch.arange(len(groups), device=groups.device) - starts[groups]
        padded = hidden.new_zeros(count, int(sizes.max().item()), hidden.shape[1])
        padded[groups, position] = hidden
        padding = torch.ones(padded.shape[:2], dtype=torch.bool, device=groups.device)
        padding[groups, position] = False
        for layer in self.attention:
            padded = layer(padded, src_key_padding_mask=padding)
        parts = [encoded, self.norm(padded[groups, position])]
        if self.similarity:
            parts.append(feed_similarity(numeric, groups, self.n_dense, self.n_vectors))
        return self.backbone(torch.cat(parts, dim=1)).squeeze(-1)


def request_layout(request_ids):
    """Row order grouping requests contiguously, plus per-request start and size."""
    _, inverse = np.unique(request_ids, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    sizes = np.bincount(inverse)
    return order, np.cumsum(sizes) - sizes, sizes


def request_batches(layout, batch_rows, generator=None):
    order, starts, sizes = layout
    requests = (
        torch.randperm(len(sizes), generator=generator).numpy()
        if generator is not None else np.arange(len(sizes))
    )
    ends = np.cumsum(sizes[requests])
    cuts = np.searchsorted(ends, np.arange(batch_rows, ends[-1], batch_rows), side="left") + 1
    for chosen in np.split(requests, np.unique(cuts[cuts < len(requests)])):
        lengths = sizes[chosen]
        before = np.repeat(np.cumsum(lengths) - lengths, lengths)
        offsets = np.repeat(starts[chosen], lengths) + np.arange(lengths.sum()) - before
        yield (
            torch.from_numpy(order[offsets]),
            torch.from_numpy(np.repeat(np.arange(len(chosen)), lengths)),
        )


def predict_feed(model, numeric, categorical, layout, device, batch_rows=8192):
    model.eval()
    probabilities = np.empty(len(layout[0]), dtype=np.float32)
    with torch.no_grad():
        for rows, groups in request_batches(layout, batch_rows):
            logits = model(numeric[rows].to(device), categorical[rows].to(device), groups.to(device))
            probabilities[rows.numpy()] = logits.sigmoid().mean(dim=1).cpu().numpy()
    return probabilities


def fit_feed(
    model, model_config, optimizer, train, valid, *, device, output, epochs=30, patience=5,
    batch_rows=1024, validation_interval=300, seed=42,
):
    (x_train, c_train, y_train, train_layout), (x_valid, c_valid, y_valid, valid_layout) = train, valid
    output.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)
    probe_requests = torch.randperm(
        len(train_layout[2]), generator=torch.Generator().manual_seed(42)
    )[: min(5000, len(train_layout[2]))].numpy()
    probe_rows = np.concatenate([
        train_layout[0][train_layout[1][r] : train_layout[1][r] + train_layout[2][r]]
        for r in np.sort(probe_requests)
    ])
    probe_layout = request_layout(np.repeat(np.arange(len(probe_requests)), train_layout[2][np.sort(probe_requests)]))
    steps_per_epoch = sum(1 for _ in request_batches(train_layout, batch_rows))
    best_auc, best_step, global_step, checks, history = -np.inf, 0, 0, [], []
    started = time.monotonic()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, rows_seen, stop = 0.0, 0, False
        for batch_index, (rows, groups) in enumerate(request_batches(train_layout, batch_rows, generator), 1):
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_train[rows].to(device), c_train[rows].to(device), groups.to(device))
            targets = y_train[rows].to(device).unsqueeze(1).expand_as(logits)
            loss = F.binary_cross_entropy_with_logits(logits, targets)
            if not torch.isfinite(loss).item():
                raise ValueError(f"Nonfinite loss at epoch {epoch}, step {global_step}")
            loss.backward()
            optimizer.step()
            global_step += 1
            loss_sum += loss.item() * len(rows)
            rows_seen += len(rows)
            expired = global_step - best_step >= patience * steps_per_epoch
            if global_step % validation_interval and batch_index != steps_per_epoch and not expired:
                continue
            probabilities = predict_feed(model, x_valid, c_valid, valid_layout, device)
            auc = float(roc_auc_score(y_valid, probabilities))
            probe = predict_feed(model, x_train[probe_rows], c_train[probe_rows], probe_layout, device)
            probe_y = y_train[probe_rows].numpy()
            record = {
                "epoch": epoch, "global_step": global_step,
                "fractional_epoch": epoch - 1 + batch_index / steps_per_epoch,
                "validation_auc": auc,
                "validation_logloss": float(log_loss(y_valid, probabilities, labels=[0, 1])),
                "training_probe_auc": float(roc_auc_score(probe_y, probe)),
                "seconds": time.monotonic() - started,
            }
            checks.append(record)
            print(json.dumps(record), flush=True)
            if auc > best_auc:
                best_auc, best_step = auc, global_step
                torch.save({
                    "model_config": model_config,
                    "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "selection": record,
                }, output / "best_model.pt")
                np.savez_compressed(
                    output / "validation_predictions.npz", y_true=y_valid, probabilities=probabilities,
                )
            model.train()
            if global_step - best_step >= patience * steps_per_epoch:
                stop = True
                break
        history.append({"epoch": epoch, "training_loss": loss_sum / rows_seen})
        if stop:
            break
    checkpoint = torch.load(output / "best_model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    return {
        "validation_auc": best_auc, "best_global_step": best_step,
        "best_fractional_epoch": checkpoint["selection"]["fractional_epoch"],
        "steps_per_epoch": steps_per_epoch, "optimizer_steps": global_step,
        "training_seconds": time.monotonic() - started,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None,
        "history": history, "checkpoint_history": checks,
    }


def request_ids_for(path, row_ids):
    return pl.scan_parquet(path).select("request_id").collect()["request_id"].to_numpy()[row_ids]


def stream_requests(path, batch_size=65536):
    """Yield (frame, file row offsets) chunks that never split a request."""
    pending, offset = None, 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size):
        frame = pl.from_arrow(batch).with_columns(
            pl.Series("_row_id", np.arange(offset, offset + batch.num_rows, dtype=np.int64)))
        offset += batch.num_rows
        if pending is not None:
            frame = pl.concat([pending, frame], how="vertical_relaxed")
        last = frame["request_id"][-1]
        split = frame.height - int((frame["request_id"] == last).sum())
        pending = frame.slice(split)
        if split:
            yield frame.slice(0, split)
    if pending is not None and pending.height:
        yield pending


def predict_file(model, path, state, device, cutoff=None):
    from custom_network import transform_custom_features
    from train import TARGET, features

    values, keys, labels = [], [], []
    for frame in stream_requests(path):
        if cutoff is not None:
            frame = frame.filter(pl.col("feature1") >= cutoff)
            if frame.is_empty():
                continue
            keys.append(frame["_row_id"].to_numpy())
            labels.append(frame[TARGET].to_numpy())
        else:
            keys.append(frame["id"].to_numpy())
        layout = request_layout(frame["request_id"].to_numpy())
        x, c = transform_custom_features(
            features(frame.drop("_row_id"), expand_embeddings=True), state)
        values.append(predict_feed(model, x, c, layout, device))
    return np.concatenate(values), np.concatenate(keys), np.concatenate(labels) if labels else None


def run(data_dir, output, spec, *, device="cuda", epochs=30, interval=300, seed=42):
    from full_experiments import adapted_encoder_config, category_view, prepare_data

    output.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "train.parquet"
    data = prepare_data(train_path, output)
    torch.manual_seed(seed)
    c_train, c_valid, cardinalities, mappings = category_view(
        data, 10, spec.get("drop_high", True), with_mappings=True)
    joblib.dump(mappings, output / "category-remappings.joblib")
    dims = [8] * len(cardinalities)
    encoder_config = adapted_encoder_config(
        data, data["x_train"].shape[1], cardinalities, dims,
        {"vector_crosses": True, "numerical_encoding": "plr"})
    config = {
        "encoder_config": encoder_config, "width": spec.get("width", 128),
        "dropout": spec.get("dropout", 0.2), "members": spec.get("members", 32),
        "positive_rate": float(data["y_train"].mean().item()),
        "heads": spec.get("heads", 4), "layers": spec.get("layers", 2),
        "attention_dropout": spec.get("attention_dropout", 0.1),
        "similarity": spec.get("similarity", True),
    }
    model = FeedAttentionClassifier(**config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=spec.get("learning_rate", 5e-4), weight_decay=spec.get("weight_decay", 0.01))
    train_layout = request_layout(request_ids_for(train_path, data["training_row_ids"]))
    valid_layout = request_layout(request_ids_for(train_path, data["validation_row_ids"]))
    print(f"START {spec['name']}: {json.dumps(spec)}; requests "
          f"{len(train_layout[2]):,}/{len(valid_layout[2]):,}", flush=True)
    metrics = fit_feed(
        model, config, optimizer,
        (data["x_train"], c_train, data["y_train"], train_layout),
        (data["x_valid"], c_valid, data["y_valid"], valid_layout),
        device=device, output=output, epochs=epochs, validation_interval=interval, seed=seed,
    )
    saved = np.load(output / "validation_predictions.npz")
    np.savez_compressed(
        output / "validation_predictions.npz", y_true=saved["y_true"],
        probabilities=saved["probabilities"], row_ids=data["validation_row_ids"])
    result = {
        "name": spec["name"], "spec": spec, "seed": seed,
        "parameters": sum(p.numel() for p in model.parameters()), "model_config": config, **metrics,
    }
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"RESULT {spec['name']}: AUC={metrics['validation_auc']:.10f}", flush=True)
    state, row_ids = {**data["state"], "category_remappings": mappings}, data["validation_row_ids"]
    del data, c_train, c_valid, optimizer
    gc.collect()
    return model, state, row_ids, result


def replay_and_submit(model, state, data_dir, output, device, cutoff):
    replay, row_ids, labels = predict_file(model, data_dir / "train.parquet", state, device, cutoff)
    with np.load(output / "validation_predictions.npz") as saved:
        position = {row: i for i, row in enumerate(row_ids.tolist())}
        order = np.array([position[row] for row in saved["row_ids"].tolist()])
        np.testing.assert_array_equal(labels[order], saved["y_true"])
        error = float(np.max(np.abs(replay[order] - saved["probabilities"])))
    auc = float(roc_auc_score(labels, replay))
    print(f"REPLAY: {len(labels):,} rows, AUC {auc:.6f}, max error {error:.2e}", flush=True)
    if error > 5e-5:
        raise ValueError(f"Holdout replay differs from saved predictions: {error}")
    template = pl.read_csv(data_dir / "sample_submission.csv")
    probabilities, ids, _ = predict_file(model, data_dir / "test.parquet", state, device)
    if not np.array_equal(ids, template["id"].to_numpy()):
        raise ValueError("Test IDs differ from the submission template")
    if not np.isfinite(probabilities).all():
        raise ValueError("Invalid test probabilities")
    path = output / "submission.csv"
    template.with_columns(pl.Series("watched", probabilities)).write_csv(path)
    report = {"holdout_replay_auc": auc, "max_replay_error": error, "rows": int(len(probabilities)),
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "refit": False}
    (output / "prediction-results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--spec", default='{"name": "feed-attention"}')
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--validation-interval", type=int, default=300)
    parser.add_argument("--cutoff", default=CUTOFF)
    parser.add_argument("--no-submission", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, state, _, _ = run(
        args.data_dir, args.output_dir, json.loads(args.spec), device=args.device,
        epochs=args.epochs, interval=args.validation_interval)
    if not args.no_submission:
        replay_and_submit(model, state, args.data_dir, args.output_dir, args.device,
                          datetime.fromisoformat(args.cutoff))


if __name__ == "__main__":
    main()
