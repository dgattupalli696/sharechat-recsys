"""LightGBM residual models on the holdout over feature subsets, starting from the S017-equivalent blend logit.

Selection: user-grouped 4-fold OOF on the first half of the holdout (by time). Confirmation: train on the
first half, score the second half. Gains are AUC(blend + residual) - AUC(blend).
"""
import csv
import json
import re
import sys

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

M = "artifacts/local-t036/matrices/"
columns = json.loads(open(M + "columns.json").read())["columns"]
x = np.load(M + "x_valid.npy", mmap_mode="r")
y = np.load(M + "y_valid.npy").astype(np.int8)
ids = np.load(M + "row_ids.npy")
z = np.load("artifacts/local-t052/teacher-blend.npz")
assert np.array_equal(z["row_ids"], ids)
p = z["probabilities"].astype(np.float64).clip(1e-6, 1 - 1e-6)
base = np.log(p / (1 - p))
meta = (pl.scan_parquet("data/train.parquet").with_row_index("_row_id")
        .filter(pl.col("_row_id").is_in(ids.tolist())).select("_row_id", "feature1", "feature5").collect())
meta = pl.DataFrame({"_row_id": ids}).join(meta, on="_row_id", how="left")
t, users = meta["feature1"].cast(pl.Int64).to_numpy(), meta["feature5"].to_numpy()
first = t < np.median(t)
A, B = np.flatnonzero(first), np.flatnonzero(~first)

level = {r["col"]: r["level"] for r in csv.DictReader(open("artifacts/local-column-profile/profile.csv"))}
groups = {}
for i, name in enumerate(columns):
    if name.startswith("rq_") and name.split("_")[1] in ("rank", "dev", "gap"):
        g = f"rq_{name.split('_')[1]}"
    elif name.startswith("cos_"):
        g = "cos"
    elif name.startswith("rq_") or name == "feature3" or level.get(name, "").endswith("cat") or level.get(name) in ("GLOBAL/TIME", None):
        g = "context"
    else:
        k = int(re.sub(r"\D", "", name))
        g = f"{'row' if level[name].startswith('ROW') else level[name].lower()}_{k // 100 * 100:03d}"
    groups.setdefault(g, []).append(i)
groups = {g: v for g, v in groups.items() if len(v) >= 3 or g == "context"}
PARAMS = dict(objective="binary", learning_rate=0.02, num_leaves=15, min_data_in_leaf=400, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=10, verbose=-1, num_threads=8)
ROUNDS = [100, 200, 400]


def fit_predict(cols, train, score):
    d = lgb.Dataset(np.ascontiguousarray(x[train][:, cols]), y[train], init_score=base[train], free_raw_data=True)
    model = lgb.train(PARAMS, d, max(ROUNDS))
    xs = np.ascontiguousarray(x[score][:, cols])
    return {r: model.predict(xs, num_iteration=r, raw_score=True) for r in ROUNDS}


def oof(cols):
    out = {r: np.zeros(len(A)) for r in ROUNDS}
    for tr, va in GroupKFold(4).split(A, groups=users[A]):
        for r, v in fit_predict(cols, A[tr], A[va]).items():
            out[r][va] = v
    a0 = roc_auc_score(y[A], base[A])
    return {r: roc_auc_score(y[A], base[A] + v) - a0 for r, v in out.items()}


def confirm(cols):
    b0 = roc_auc_score(y[B], base[B])
    return {r: roc_auc_score(y[B], base[B] + v) - b0 for r, v in fit_predict(cols, A, B).items()}


def main():
    rng = np.random.default_rng(0)
    allc = list(range(len(columns)))
    cands = {"all": allc, **{f"only:{g}": v for g, v in groups.items()},
             **{f"drop:{g}": [c for c in allc if c not in set(v)] for g, v in groups.items()},
             **{f"rand30:{s}": sorted(rng.choice(len(columns), len(columns) * 3 // 10, replace=False)) for s in range(8)}}
    print(f"holdout A {len(A)} B {len(B)}; base A {roc_auc_score(y[A], base[A]):.5f} B {roc_auc_score(y[B], base[B]):.5f};"
          f" {len(groups)} groups: " + ", ".join(f"{g}({len(v)})" for g, v in groups.items()), flush=True)
    rows = []
    for name, cols in cands.items():
        gain = oof(cols)
        best = max(gain, key=gain.get)
        rows.append((gain[best], name, best))
        print(f"{name:22s} n={len(cols):3d} oofA " + " ".join(f"{r}:{g:+.5f}" for r, g in gain.items()), flush=True)
    rows.sort(reverse=True)
    print("\nconfirm top 6 on B (train A):")
    for g, name, r in rows[:6]:
        c = confirm(cands[name])
        print(f"{name:22s} oofA {g:+.5f}@{r}  B " + " ".join(f"{k}:{v:+.5f}" for k, v in c.items()), flush=True)


if __name__ == "__main__":
    main()
