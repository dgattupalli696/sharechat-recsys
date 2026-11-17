"""T050: multi-tower network on the T036 holdout matrices.

Towers see different views of each row and are fused by a gated MLP:
  content - the 787 T036 dense columns, quantile-binned (bin embedding + cdf scalar + missing flag)
  id      - learned embeddings of raw item/creator/user IDs (frequent values only) + pairwise dots
  leaf    - embeddings of XGBoost depth12 leaf indices (trees pooled in groups), GBDT interactions
Each tower has an auxiliary head and whole-tower dropout so weak towers still learn.

`multitower_prepare.py` (after `xgb_tune.py build --cosines`) writes binned matrices, ID codes
and leaf indices; this module trains one variant on one device. The two are separate processes
because torch and xgboost OpenMP runtimes segfault together on macOS.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from multitower_prepare import BINS

TOWERS = ("content", "id", "leaf")


class MultiTower(nn.Module):
    def __init__(self, n_dense, id_cards, n_trees, leaf_nodes, towers=TOWERS, width=256, bin_dim=8,
                 id_dim=32, leaf_dim=16, leaf_groups=16, dropout=0.2, tower_dropout=0.1, prior=0.05):
        super().__init__()
        self.towers, self.tower_dropout, self.leaf_nodes = tuple(towers), tower_dropout, leaf_nodes
        blocks, heads = nn.ModuleDict(), nn.ModuleDict()

        def mlp(inputs, hidden):
            return nn.Sequential(nn.Linear(inputs, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, width), nn.LayerNorm(width))

        if "content" in towers:
            self.bin_embedding = nn.Parameter(torch.randn(n_dense, BINS + 2, bin_dim) * 0.05)
            blocks["content"] = mlp(n_dense * (bin_dim + 2), 1024)
        if "id" in towers:
            self.id_embeddings = nn.ModuleList(nn.Embedding(c, id_dim) for c in id_cards)
            pairs = len(id_cards) * (len(id_cards) - 1) // 2
            blocks["id"] = mlp(len(id_cards) * id_dim + pairs, 512)
        if "leaf" in towers:
            self.leaf_embedding = nn.Embedding(n_trees * leaf_nodes, leaf_dim)
            nn.init.normal_(self.leaf_embedding.weight, std=0.01)
            groups = torch.arange(n_trees) * leaf_groups // n_trees
            assign = F.one_hot(groups, leaf_groups).float()
            self.register_buffer("leaf_assign", assign / assign.sum(0, keepdim=True))
            self.register_buffer("leaf_offsets", torch.arange(n_trees) * leaf_nodes)
            blocks["leaf"] = mlp(leaf_groups * leaf_dim, 512)
        for name in towers:
            heads[name] = nn.Linear(width, 1)
        self.blocks, self.heads = blocks, heads
        self.gate = nn.Linear(width * len(towers), len(towers))
        self.fusion = nn.Sequential(nn.Linear(width * len(towers), 512), nn.GELU(), nn.Dropout(dropout),
                                    nn.Linear(512, 1))
        bias = float(np.log(prior / (1 - prior)))
        for head in [*heads.values(), self.fusion[-1]]:
            nn.init.constant_(head.bias, bias)

    def tower_inputs(self, bins, ids, leaves):
        out = {}
        if "content" in self.towers:
            missing = bins < 0
            index = torch.where(missing, torch.full_like(bins, BINS + 1), bins).long()
            column = torch.arange(bins.shape[1], device=bins.device)
            embedded = self.bin_embedding[column, index]
            cdf = torch.where(missing, 0.0, index.float() / BINS * 2 - 1)
            out["content"] = torch.cat([embedded.flatten(1), cdf, missing.float()], 1)
        if "id" in self.towers:
            vectors = [e(ids[:, i].long()) for i, e in enumerate(self.id_embeddings)]
            dots = [(vectors[i] * vectors[j]).sum(1, keepdim=True)
                    for i in range(len(vectors)) for j in range(i + 1, len(vectors))]
            out["id"] = torch.cat([*vectors, *dots], 1)
        if "leaf" in self.towers:
            embedded = self.leaf_embedding(leaves.long() + self.leaf_offsets)
            out["leaf"] = torch.einsum("btd,tg->bgd", embedded, self.leaf_assign).flatten(1)
        return out

    def forward(self, bins, ids, leaves):
        inputs = self.tower_inputs(bins, ids, leaves)
        reps = [self.blocks[name](inputs[name]) for name in self.towers]
        aux = torch.cat([self.heads[name](r) for name, r in zip(self.towers, reps)], 1)
        if self.training and self.tower_dropout > 0 and len(reps) > 1:
            keep = (torch.rand(reps[0].shape[0], len(reps), device=reps[0].device) > self.tower_dropout).float()
            keep[keep.sum(1) == 0, 0] = 1.0
            reps = [r * keep[:, i:i + 1] for i, r in enumerate(reps)]
        gates = torch.softmax(self.gate(torch.cat(reps, 1)), 1) * len(reps)
        fused = torch.cat([r * gates[:, i:i + 1] for i, r in enumerate(reps)], 1)
        return self.fusion(fused).squeeze(1), aux


def predict(model, tensors, batch=8192):
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast(tensors[0].device.type, dtype=torch.float16,
                                         enabled=tensors[0].device.type == "cuda"):
        for start in range(0, len(tensors[0]), batch):
            logits, _ = model(*(t[start:start + batch] for t in tensors))
            out.append(torch.sigmoid(logits.float()).cpu())
    return torch.cat(out).numpy()


def fit(model, train, y_train, valid, y_valid, *, epochs=12, batch=2048, lr=1e-3, aux_weight=0.2,
        evals_per_epoch=2, patience=4, seed=42, log=print):
    """AdamW with warmup + cosine; keeps the best holdout-AUC state. Returns (best_auc, predictions, history)."""
    from sklearn.metrics import roc_auc_score

    device = train[0].device
    generator = torch.Generator(device="cpu").manual_seed(seed)
    sparse = [p for n, p in model.named_parameters() if "embedding" in n]
    dense = [p for n, p in model.named_parameters() if "embedding" not in n]
    optimizer = torch.optim.AdamW([{"params": dense, "weight_decay": 1e-5},
                                   {"params": sparse, "weight_decay": 0.0}], lr=lr)
    steps_per_epoch = (len(y_train) + batch - 1) // batch
    total = epochs * steps_per_epoch
    warmup = min(500, total // 10)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1, (s + 1) / warmup) * (
        0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * min(1.0, s / total)))))
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    interval = max(1, steps_per_epoch // evals_per_epoch)
    best, best_state, best_pred, history, bad, step = -1.0, None, None, [], 0, 0
    y_valid_np = y_valid.cpu().numpy()
    began = time.time()
    for epoch in range(epochs):
        order = torch.randperm(len(y_train), generator=generator).to(device)
        model.train()
        for start in range(0, len(order), batch):
            index = order[start:start + batch]
            target = y_train[index].float()
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits, aux = model(*(t[index] for t in train))
            loss = F.binary_cross_entropy_with_logits(logits.float(), target)
            loss = loss + aux_weight * F.binary_cross_entropy_with_logits(
                aux.float(), target[:, None].expand_as(aux))
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            schedule.step()
            step += 1
            if step % interval == 0 or step == total:
                pred = predict(model, valid)
                auc = float(roc_auc_score(y_valid_np, pred))
                history.append({"step": step, "epoch": round(step / steps_per_epoch, 2), "auc": auc,
                                "loss": float(loss.detach()), "seconds": round(time.time() - began)})
                log(json.dumps(history[-1]))
                if auc > best:
                    best, best_pred, bad = auc, pred, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
                model.train()
                if bad >= patience:
                    break
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    return best, best_pred, history


def train_variant(args):
    m, p = args.matrices, args.prepared
    meta = json.loads((p / "meta.json").read_text())
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    def load(split):
        arrays = [np.load(p / f"bins_{split}.npy"), np.load(p / f"ids_{split}.npy"), np.load(p / f"leaves_{split}.npy")]
        return [torch.from_numpy(a).to(device) for a in arrays]

    train, valid = load("fit"), load("valid")
    y_train = torch.from_numpy(np.load(m / "y_fit.npy").astype(np.float32)).to(device)
    y_valid = torch.from_numpy(np.load(m / "y_valid.npy").astype(np.float32))
    towers = args.towers.split(",")
    model = MultiTower(meta["n_dense"], meta["id_cards"], meta["trees"], meta["leaf_nodes"], towers=towers,
                       prior=float(y_train.mean())).to(device)
    print(args.name, "towers", towers, "params", sum(q.numel() for q in model.parameters()), flush=True)
    auc, pred, history = fit(model, train, y_train, valid, y_valid, epochs=args.epochs, batch=args.batch,
                             lr=args.lr, seed=args.seed, log=lambda s: print(args.name, s, flush=True))
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / f"{args.name}-holdout.npz", row_ids=np.load(m / "row_ids.npy"),
                        y_true=y_valid.numpy(), probabilities=pred.astype(np.float32))
    with torch.no_grad():
        model.eval()
        sample = [t[:65536] for t in valid]
        _, aux = model(*sample)
        aux_auc = {}
        from sklearn.metrics import roc_auc_score
        for i, name in enumerate(towers):
            aux_auc[name] = float(roc_auc_score(y_valid[:65536].numpy(), aux[:, i].float().cpu().numpy()))
    result = {"name": args.name, "towers": towers, "holdout_auc": auc, "aux_auc_first64k": aux_auc,
              "history": history}
    (args.output / f"{args.name}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(args.name, json.dumps({"holdout_auc": auc, "aux": aux_auc}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrices", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--name", default="full")
    parser.add_argument("--towers", default=",".join(TOWERS))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_variant(args)


if __name__ == "__main__":
    main()
