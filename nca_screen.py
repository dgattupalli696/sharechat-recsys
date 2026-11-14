"""ModernNCA-style retrieval screen on the T036 holdout matrices.

Rows are embedded by a small MLP over categorical/ID embeddings plus standardised
dense features. A row's prediction is the softmax(-squared distance) weighted mean
label of training candidates. Training samples a random candidate pool per batch
(excluding the batch's own rows); evaluation retrieves against every fit row.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from deepfm_screen import prepare


class Encoder(nn.Module):
    def __init__(self, cardinalities, n_dense, dim=8, hidden=512, out=128, dropout=0.15):
        super().__init__()
        self.embeddings = nn.ModuleList(nn.Embedding(n, dim) for n in cardinalities)
        for e in self.embeddings:
            nn.init.normal_(e.weight, std=0.01)
        self.net = nn.Sequential(nn.Linear(len(cardinalities) * dim + n_dense, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, out))

    def forward(self, codes, dense):
        cat = torch.cat([e(codes[:, i]) for i, e in enumerate(self.embeddings)], 1)
        return self.net(torch.cat([cat, dense], 1))


def neighbour_probability(query, keys, labels, temperature, exclude=None):
    distance = torch.cdist(query, keys).pow(2) / temperature
    if exclude is not None:
        distance = distance.masked_fill(exclude, float("inf"))
    weights = torch.softmax(-distance, 1)
    return (weights * labels[None, :]).sum(1)


def encode(model, codes, dense, device, batch=16384):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(dense), batch):
            out.append(model(torch.from_numpy(codes[start:start + batch]).to(device),
                             torch.from_numpy(dense[start:start + batch]).to(device)))
    return torch.cat(out)


def retrieve(query, keys, labels, temperature, query_batch=1024, key_batch=262144):
    """Streaming softmax over all keys: returns sum(w*y)/sum(w) per query."""
    out, key_norm = [], keys.pow(2).sum(1)
    for qs in range(0, len(query), query_batch):
        q = query[qs:qs + query_batch]
        running_max = torch.full((len(q),), -float("inf"), device=q.device)
        numerator = torch.zeros(len(q), device=q.device)
        denominator = torch.zeros(len(q), device=q.device)
        q_norm = q.pow(2).sum(1, keepdim=True)
        for ks in range(0, len(keys), key_batch):
            block = keys[ks:ks + key_batch]
            logits = -(q_norm + key_norm[None, ks:ks + key_batch] - 2 * q @ block.T).clamp_min(0) / temperature
            block_max = logits.max(1).values
            new_max = torch.maximum(running_max, block_max)
            scale = torch.exp(running_max - new_max)
            w = torch.exp(logits - new_max[:, None])
            numerator = numerator * scale + (w * labels[None, ks:ks + key_batch]).sum(1)
            denominator = denominator * scale + w.sum(1)
            running_max = new_max
        out.append((numerator / denominator).cpu())
    return torch.cat(out).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--matrices", type=Path, default=Path("artifacts/local-t036/matrices"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/local-t043"))
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--minimum", type=int, default=3)
    parser.add_argument("--rows", type=int, default=0)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--candidates", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-sample", type=int, default=50_000)
    parser.add_argument("--name", default="modernnca")
    args = parser.parse_args()
    torch.manual_seed(42)
    started = time.time()
    codes_fit, codes_valid, cardinalities, d_fit, d_valid, y_fit, y_valid, row_ids = prepare(args, started)
    device = args.device
    model = Encoder(cardinalities, d_fit.shape[1]).to(device)
    # Learnable temperature (log-parameterised), shared across the whole retrieval.
    log_temperature = nn.Parameter(torch.tensor(0.0, device=device))
    optimizer = torch.optim.AdamW([{"params": model.parameters(), "weight_decay": 1e-5},
                                   {"params": [log_temperature], "weight_decay": 0.0}], lr=args.lr)
    labels_fit = torch.from_numpy(y_fit.astype(np.float32)).to(device)
    generator = torch.Generator().manual_seed(42)
    best, history = (-1.0, None, 0), []
    args.output.mkdir(parents=True, exist_ok=True)
    n = len(y_fit)
    sample = np.sort(np.random.default_rng(7).choice(len(y_valid), min(args.eval_sample, len(y_valid)), replace=False))
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(n, generator=generator).numpy()
        began, total = time.time(), 0.0
        for start in range(0, n, args.batch):
            idx = np.sort(order[start:start + args.batch])
            pool = np.sort(np.random.default_rng(epoch * 1_000_003 + start).choice(n, args.candidates, replace=False))
            rows = np.concatenate([idx, pool])
            embedded = model(torch.from_numpy(codes_fit[rows]).to(device), torch.from_numpy(d_fit[rows]).to(device))
            query, keys = embedded[:len(idx)], embedded[len(idx):]
            exclude = torch.from_numpy(idx[:, None] == pool[None, :]).to(device)
            p = neighbour_probability(query, keys, labels_fit[torch.from_numpy(pool).to(device)],
                                      log_temperature.exp(), exclude).clamp(1e-6, 1 - 1e-6)
            y = labels_fit[torch.from_numpy(idx).to(device)]
            loss = -(y * p.log() + (1 - y) * (1 - p).log()).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(idx)
        keys = encode(model, codes_fit, d_fit, device)
        query = encode(model, codes_valid[sample], d_valid[sample], device)
        probabilities = retrieve(query, keys, labels_fit, log_temperature.detach().exp())
        del keys, query
        auc = float(roc_auc_score(y_valid[sample], probabilities))
        history.append({"epoch": epoch, "loss": total / n, "sample_auc": auc,
                        "temperature": log_temperature.detach().exp().item(), "seconds": time.time() - began})
        print(json.dumps(history[-1]), flush=True)
        if auc > best[0]:
            best = (auc, {k: v.detach().clone() for k, v in model.state_dict().items()}, epoch,
                    log_temperature.detach().clone())
        elif epoch - best[2] >= 2:
            break
    model.load_state_dict(best[1])
    keys = encode(model, codes_fit, d_fit, device)
    query = encode(model, codes_valid, d_valid, device)
    probabilities = retrieve(query, keys, labels_fit, best[3].exp())
    holdout_auc = float(roc_auc_score(y_valid, probabilities))
    np.savez_compressed(args.output / f"{args.name}-holdout.npz", row_ids=row_ids, y_true=y_valid,
                        probabilities=probabilities.astype(np.float32))
    torch.save({"model": best[1], "log_temperature": best[3]}, args.output / f"{args.name}.pt")
    result = {"holdout_auc": holdout_auc, "best_epoch": best[2], "history": history, "seconds": time.time() - started}
    (args.output / f"{args.name}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(args.name, json.dumps({k: result[k] for k in ["holdout_auc", "best_epoch", "seconds"]}), flush=True)


if __name__ == "__main__":
    main()
