"""T042: DeepFM with user/item ID embeddings on the T036 holdout matrices.

Our TabM/DCN models drop high-cardinality fields; this model embeds user, item and
other ID fields (fit-period vocabulary, rare values pooled) alongside the 770 dense
features. Logit = linear + FM pairwise term over field embeddings + MLP. Original
holdout protocol; best epoch selected on the holdout; predictions saved for blending.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
import torch
from torch import nn

from xgb_tune import ID_COLUMNS, train_split_columns


def id_codes(data, rows, minimum):
    fit, valid = train_split_columns(data, ID_COLUMNS, rows)
    out_fit, out_valid, sizes = [], [], []
    for c in ID_COLUMNS:
        counts = fit[c].fill_null("__NA__").value_counts()
        keep = counts.filter(pl.col("count") >= minimum)[c]
        mapping = pl.DataFrame({c: keep, "code": np.arange(1, len(keep) + 1, dtype=np.int64)})
        for frame, out in [(fit, out_fit), (valid, out_valid)]:
            joined = frame.select(pl.col(c).fill_null("__NA__")).join(mapping, on=c, how="left", maintain_order="left")
            out.append(joined["code"].fill_null(0).to_numpy())
        sizes.append(len(keep) + 1)
    return np.column_stack(out_fit), np.column_stack(out_valid), sizes


def dense_transform(x_fit, x_valid, dense):
    sample = np.asarray(x_fit[:: max(1, len(x_fit) // 200_000)][:, dense])
    low, high = np.nanpercentile(sample, [0.5, 99.5], axis=0)
    clipped = np.clip(sample, low, high)
    mean, std = np.nanmean(clipped, axis=0), np.nanstd(clipped, axis=0) + 1e-6
    mean, std = np.nan_to_num(mean), np.nan_to_num(std, nan=1.0)

    def apply(x):
        out = np.empty((len(x), len(dense)), dtype=np.float32)
        for start in range(0, len(x), 100_000):
            v = np.asarray(x[start:start + 100_000][:, dense], dtype=np.float32)
            v = (np.clip(v, low, high) - mean) / std
            out[start:start + len(v)] = np.nan_to_num(v, nan=0.0)
        return out

    return apply(x_fit), apply(x_valid)


class DeepFM(nn.Module):
    def __init__(self, cardinalities, n_dense, dim=16, numeric_fields=16, hidden=(512, 256, 128), dropout=0.2):
        super().__init__()
        self.embeddings = nn.ModuleList(nn.Embedding(n, dim) for n in cardinalities)
        self.linear = nn.ModuleList(nn.Embedding(n, 1) for n in cardinalities)
        for e in [*self.embeddings, *self.linear]:
            nn.init.normal_(e.weight, std=0.01)
        self.numeric_fields = nn.Linear(n_dense, numeric_fields * dim)
        nn.init.normal_(self.numeric_fields.weight, std=0.01 / np.sqrt(n_dense))
        nn.init.zeros_(self.numeric_fields.bias)
        self.bias = nn.Parameter(torch.tensor([-2.944]))
        self.numeric_linear = nn.Linear(n_dense, 1)
        self.numeric_norm = nn.LayerNorm(dim)
        self.fm_scale = 1.0 / (len(cardinalities) + numeric_fields)
        self.dim, self.k = dim, numeric_fields
        layers, width = [], len(cardinalities) * dim + n_dense
        for h in hidden:
            layers += [nn.Linear(width, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            width = h
        self.mlp = nn.Sequential(*layers, nn.Linear(width, 1))

    def forward(self, codes, dense):
        fields = [e(codes[:, i]) for i, e in enumerate(self.embeddings)]
        cat = torch.stack(fields, 1)
        num = 0.1 * self.numeric_norm(self.numeric_fields(dense).view(len(dense), self.k, self.dim))
        v = torch.cat([cat, num], 1)
        fm = self.fm_scale * 0.5 * (v.sum(1).pow(2) - v.pow(2).sum(1)).sum(1, keepdim=True)
        linear = sum(l(codes[:, i]) for i, l in enumerate(self.linear)) + self.numeric_linear(dense)
        deep = self.mlp(torch.cat([cat.flatten(1), dense], 1))
        return (self.bias + linear + fm + deep).squeeze(1)


def predict(model, codes, dense, device, batch=16384):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(dense), batch):
            out.append(torch.sigmoid(model(torch.from_numpy(codes[start:start + batch]).to(device),
                                           torch.from_numpy(dense[start:start + batch]).to(device))).cpu().numpy())
    return np.concatenate(out)


def prepare(args, started):
    m = args.matrices
    meta = json.loads((m / "columns.json").read_text())
    x_fit, x_valid = np.load(m / "x_fit.npy", mmap_mode="r"), np.load(m / "x_valid.npy", mmap_mode="r")
    y_fit, y_valid, row_ids = np.load(m / "y_fit.npy"), np.load(m / "y_valid.npy"), np.load(m / "row_ids.npy")
    categorical = meta["categorical"]
    dense = [i for i in range(len(meta["columns"])) if i not in set(categorical)]
    low_fit = np.nan_to_num(np.asarray(x_fit[:, categorical]), nan=-1).astype(np.int64) + 1
    low_valid = np.nan_to_num(np.asarray(x_valid[:, categorical]), nan=-1).astype(np.int64) + 1
    low_valid = np.where(low_valid <= low_fit.max(0), low_valid, 0)
    id_fit, id_valid, id_sizes = id_codes(args.data, args.rows, args.minimum)
    codes_fit, codes_valid = np.hstack([low_fit, id_fit]), np.hstack([low_valid, id_valid])
    cardinalities = [int(v) + 1 for v in low_fit.max(0)] + id_sizes
    d_fit, d_valid = dense_transform(x_fit, x_valid, dense)
    print(f"prepared {d_fit.shape} codes {cardinalities} in {time.time() - started:.0f}s", flush=True)
    return codes_fit, codes_valid, cardinalities, d_fit, d_valid, y_fit, y_valid, row_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--matrices", type=Path, default=Path("artifacts/local-t036/matrices"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/local-t042"))
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--minimum", type=int, default=3)
    parser.add_argument("--rows", type=int, default=0)
    parser.add_argument("--name", default="deepfm")
    args = parser.parse_args()
    torch.manual_seed(42)
    started = time.time()
    codes_fit, codes_valid, cardinalities, d_fit, d_valid, y_fit, y_valid, row_ids = prepare(args, started)
    model = DeepFM(cardinalities, d_fit.shape[1]).to(args.device)
    embed = [p for n, p in model.named_parameters() if "embeddings" in n or n.startswith("linear.")]
    other = [p for n, p in model.named_parameters() if not ("embeddings" in n or n.startswith("linear."))]
    optimizer = torch.optim.AdamW([{"params": embed, "weight_decay": 0.0},
                                   {"params": other, "weight_decay": 0.01}], lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(42)
    best, history = (-1.0, None, 0), []
    args.output.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(y_fit), generator=generator).numpy()
        began, total = time.time(), 0.0
        for start in range(0, len(order), 2048):
            idx = np.sort(order[start:start + 2048])
            c = torch.from_numpy(codes_fit[idx]).to(args.device)
            d = torch.from_numpy(d_fit[idx]).to(args.device)
            y = torch.from_numpy(y_fit[idx].astype(np.float32)).to(args.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(c, d)
            loss = loss_fn(logits, y)
            # L2 only on the embedding rows used in this batch (sparse-friendly regularization)
            reg = sum(e(c[:, i]).pow(2).sum() for i, e in enumerate(model.embeddings)) * 1e-6 / len(idx)
            (loss + reg).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(idx)
        probabilities = predict(model, codes_valid, d_valid, args.device)
        auc = float(roc_auc_score(y_valid, probabilities))
        history.append({"epoch": epoch, "loss": total / len(order), "holdout_auc": auc, "seconds": time.time() - began})
        print(json.dumps(history[-1]), flush=True)
        if auc > best[0]:
            best = (auc, probabilities, epoch)
        elif epoch - best[2] >= 2:
            break
    np.savez_compressed(args.output / f"{args.name}-holdout.npz", row_ids=row_ids, y_true=y_valid,
                        probabilities=best[1].astype(np.float32))
    result = {"holdout_auc": best[0], "best_epoch": best[2], "history": history,
              "cardinalities": cardinalities, "seconds": time.time() - started}
    (args.output / f"{args.name}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(args.name, json.dumps({k: result[k] for k in ["holdout_auc", "best_epoch", "seconds"]}), flush=True)


if __name__ == "__main__":
    main()
