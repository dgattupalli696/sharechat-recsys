"""T044 exposure-graph SVD screen.

Builds label-free bipartite exposure graphs over train+test (user x item, user x creator),
takes a degree-normalised truncated SVD, and adds per-row user/entity factors, their
affinity (dot product) and its within-request rank/deviation to the T036 holdout
matrices. LightGBM control vs +svd on the original 307,528-row holdout.
"""
import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score

from feature_screen import CUTOFF

USER = "feature5"
GRAPHS = {"item": "feature18", "creator": "feature23"}


def graph_factors(users, entities, k, seed=42):
    u_codes, u_index = np.unique(users, return_inverse=True)
    e_codes, e_index = np.unique(entities, return_inverse=True)
    a = sp.coo_matrix((np.ones(len(u_index), dtype=np.float32), (u_index, e_index)),
                      shape=(len(u_codes), len(e_codes))).tocsr()
    a.data[:] = 1.0  # binary exposure
    du = np.asarray(a.sum(1)).ravel()
    de = np.asarray(a.sum(0)).ravel()
    norm = sp.diags(1 / np.sqrt(du)) @ a @ sp.diags(1 / np.sqrt(de))
    svd = TruncatedSVD(k, algorithm="randomized", n_iter=5, random_state=seed)
    user_factors = svd.fit_transform(norm).astype(np.float32)  # U * S
    entity_factors = svd.components_.T.astype(np.float32)  # V
    return u_index, e_index, user_factors, entity_factors


def svd_features(data, k, keep):
    cols = ["request_id", "feature1", USER, *GRAPHS.values()]
    train = pl.scan_parquet(data / "train.parquet").select(cols).collect()
    test = pl.scan_parquet(data / "test.parquet").select(cols).collect()
    both = pl.concat([train, test])
    n_train = len(train)
    extra = {}
    for name, column in GRAPHS.items():
        started = time.time()
        u_idx, e_idx, uf, ef = graph_factors(both[USER].to_numpy(), both[column].to_numpy(), k)
        u, e = uf[u_idx[:n_train]], ef[e_idx[:n_train]]
        extra[f"svd_{name}_dot"] = (u * e).sum(1)
        cos = extra[f"svd_{name}_dot"] / (np.linalg.norm(u, axis=1) * np.linalg.norm(e, axis=1) + 1e-9)
        extra[f"svd_{name}_cos"] = cos.astype(np.float32)
        extra[f"svd_{name}_unorm"] = np.linalg.norm(u, axis=1)
        extra[f"svd_{name}_enorm"] = np.linalg.norm(e, axis=1)
        for j in range(keep):
            extra[f"svd_{name}_u{j}"] = u[:, j]
            extra[f"svd_{name}_e{j}"] = e[:, j]
        print(f"svd {name}: users {uf.shape[0]} entities {ef.shape[0]} in {time.time() - started:.0f}s", flush=True)
    frame = pl.DataFrame({"request_id": train["request_id"], **extra})
    pairs = [c for c in extra if c.endswith(("_dot", "_cos"))]
    frame = frame.with_columns([
        e for c in pairs for e in (
            (pl.col(c).rank("average").over("request_id") / pl.len().over("request_id")).alias(f"rq_rank_{c}"),
            (pl.col(c) - pl.col(c).mean().over("request_id")).alias(f"rq_dev_{c}"))]).drop("request_id")
    is_valid = (train["feature1"] >= CUTOFF).to_numpy()
    values = frame.to_numpy().astype(np.float32)
    return values[~is_valid], values[is_valid], frame.columns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--matrices", type=Path, default=Path("artifacts/local-t036/matrices"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/local-t044"))
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--keep", type=int, default=8, help="Raw factor components kept per side")
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--variants", default="control,svd")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    s_fit, s_valid, s_names = svd_features(args.data, args.k, args.keep)
    m = args.matrices
    meta = json.loads((m / "columns.json").read_text())
    x_fit, x_valid = np.load(m / "x_fit.npy", mmap_mode="r"), np.load(m / "x_valid.npy", mmap_mode="r")
    y_fit, y_valid = np.load(m / "y_fit.npy"), np.load(m / "y_valid.npy")
    assert len(s_fit) == len(y_fit) and len(s_valid) == len(y_valid)
    univariate = {}
    for j, c in enumerate(s_names):
        v = s_valid[:, j]
        auc = roc_auc_score(y_valid, np.nan_to_num(v))
        univariate[c] = round(float(max(auc, 1 - auc)), 4)
    print("univariate:", json.dumps(dict(sorted(univariate.items(), key=lambda p: -p[1])[:15])), flush=True)
    results = {}
    for variant in args.variants.split(","):
        began = time.time()
        columns = meta["columns"] + (s_names if variant == "svd" else [])
        fit = np.hstack([x_fit, s_fit]) if variant == "svd" else np.asarray(x_fit)
        valid = np.hstack([x_valid, s_valid]) if variant == "svd" else np.asarray(x_valid)
        train_set = lgb.Dataset(fit, y_fit, categorical_feature=meta["categorical"], free_raw_data=True)
        valid_set = lgb.Dataset(valid, y_valid, reference=train_set)
        params = {"objective": "binary", "metric": "auc", "learning_rate": args.learning_rate,
                  "num_leaves": 127, "min_data_in_leaf": 200, "feature_fraction": 0.5,
                  "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10, "max_bin": 127,
                  "seed": 42, "num_threads": args.threads, "verbosity": -1,
                  "cat_smooth": 20, "max_cat_to_onehot": 8}
        model = lgb.train(params, train_set, 5000, valid_sets=[valid_set], valid_names=["holdout"],
                          callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(250)])
        prediction = model.predict(valid, num_iteration=model.best_iteration)
        gain = model.feature_importance("gain")
        top_svd = sorted(((columns[i], float(gain[i])) for i in range(len(meta["columns"]), len(columns))),
                         key=lambda p: -p[1])[:10]
        results[variant] = {"holdout_auc": float(roc_auc_score(y_valid, prediction)),
                            "best_iteration": model.best_iteration, "seconds": time.time() - began,
                            "svd_gain_rank": [[c, g, int((gain > g).sum()) + 1] for c, g in top_svd]}
        np.save(args.output / f"valid-{variant}.npy", prediction.astype(np.float32))
        print(variant, json.dumps(results[variant]), flush=True)
        del fit, valid, train_set, valid_set, model
    (args.output / "results.json").write_text(json.dumps(
        {"k": args.k, "keep": args.keep, "univariate": univariate, "results": results,
         "seconds": time.time() - started}, indent=2) + "\n")


if __name__ == "__main__":
    main()
