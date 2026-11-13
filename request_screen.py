"""T026 local LightGBM screen: wider within-request context features.

Same protocol as T023 (train before the original cutoff, score the original
307,528-row holdout, early stopping on the holdout for every variant, so deltas
are comparable). Wide features are chosen by gain from a fixed-budget base
model fitted on training rows only; the holdout is never used to choose them.
"""
import argparse
import gc
import json
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from feature_screen import CUTOFF, ITEM, LOW_CATEGORICAL, RANKED, scalar_numeric

PARAMS = {
    "objective": "binary", "metric": "auc", "num_leaves": 127, "min_data_in_leaf": 200,
    "feature_fraction": 0.5, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10,
    "max_bin": 127, "seed": 42, "verbosity": -1, "cat_smooth": 20, "max_cat_to_onehot": 8,
}


def context(columns, stats=()):
    group = "request_id"
    size = pl.len().over(group)
    exprs = [size.alias("rq_size"), pl.col(ITEM).n_unique().over(group).alias("rq_items"),
             pl.len().over([group, ITEM]).alias("rq_item_dup")]
    for c in columns:
        exprs += [(pl.col(c).rank("average").over(group) / size).alias(f"rq_rank_{c}"),
                  (pl.col(c) - pl.col(c).mean().over(group)).alias(f"rq_dev_{c}"),
                  (pl.col(c) - pl.col(c).max().over(group)).alias(f"rq_gap_{c}")]
    for c in stats:
        exprs += [pl.col(c).mean().over(group).alias(f"rq_mean_{c}"),
                  pl.col(c).std().over(group).alias(f"rq_std_{c}")]
    return [e.cast(pl.Float32) for e in exprs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/local-t026"))
    parser.add_argument("--variants", default="control,no_size,wide,wide_stats")
    parser.add_argument("--wide", type=int, default=50)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--rows", type=int, default=0, help="Keep 1/N of requests for smoke tests")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.time()
    numeric = scalar_numeric(pl.scan_parquet(args.data / "train.parquet").collect_schema())
    frame = pl.scan_parquet(args.data / "train.parquet")
    if args.rows:
        frame = frame.filter(pl.col("request_id").hash(7) % args.rows == 0)
    frame = frame.select("watched", "feature1", "feature3", "request_id", ITEM,
                         *dict.fromkeys([*LOW_CATEGORICAL, *numeric])).collect()
    is_valid = (frame["feature1"] >= CUTOFF).to_numpy()
    categories = {c: frame.filter(pl.Series(~is_valid))[c].drop_nulls().unique().sort().to_list()
                  for c in LOW_CATEGORICAL}
    base = ["feature3", *numeric, *LOW_CATEGORICAL]

    def matrix(source, columns):
        parts = []
        for c in columns:
            column = pl.col(c)
            if c in categories:
                column = column.cast(pl.Enum(categories[c]), strict=False).to_physical()
            parts.append(source.select(column.cast(pl.Float32))[c].to_numpy())
        return np.ascontiguousarray(np.column_stack(parts), dtype=np.float32)

    y = frame["watched"].to_numpy()
    y_fit, y_valid = y[~is_valid], y[is_valid]
    categorical = lambda columns: [i for i, c in enumerate(columns) if c in categories]
    x = matrix(frame.filter(pl.Series(~is_valid)), base)
    selector = lgb.train({**PARAMS, "learning_rate": .05, "num_threads": args.threads},
                         lgb.Dataset(x, y_fit, feature_name=base, categorical_feature=categorical(base)), 150)
    gains = dict(zip(base, selector.feature_importance("gain")))
    del x, selector
    gc.collect()
    candidates = [c for c in numeric if c not in categories and c not in {"feature3"}]
    wide = sorted(candidates, key=lambda c: -gains[c])[:args.wide]
    wide = list(dict.fromkeys([*RANKED, *wide]))
    print(f"wide context over {len(wide)} features; top: {wide[10:20]}", flush=True)
    frame = frame.with_columns(context(wide, wide[:10]))
    control = ["rq_size", "rq_items", "rq_item_dup"] + [f"rq_{k}_{c}" for c in RANKED for k in ["rank", "dev", "gap"]]
    wide_names = ["rq_size", "rq_items", "rq_item_dup"] + [f"rq_{k}_{c}" for c in wide for k in ["rank", "dev", "gap"]]
    groups = {
        "control": base + control,
        "no_size": base + [c for c in control if c != "rq_size"],
        "wide": base + wide_names,
        "wide_stats": base + wide_names + [f"rq_{k}_{c}" for c in wide[:10] for k in ["mean", "std"]],
    }
    fit_rows, valid_rows = frame.filter(pl.Series(~is_valid)), frame.filter(pl.Series(is_valid))
    del frame
    gc.collect()
    results = {}
    for name in args.variants.split(","):
        columns, began = groups[name], time.time()
        train_set = lgb.Dataset(matrix(fit_rows, columns), y_fit, feature_name=columns,
                                categorical_feature=categorical(columns), free_raw_data=True)
        x_valid = matrix(valid_rows, columns)
        valid_set = lgb.Dataset(x_valid, y_valid, reference=train_set)
        model = lgb.train({**PARAMS, "learning_rate": .05, "num_threads": args.threads}, train_set, 4000,
                          valid_sets=[valid_set], valid_names=["holdout"],
                          callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(250)])
        auc = float(roc_auc_score(y_valid, model.predict(x_valid, num_iteration=model.best_iteration)))
        top = sorted(zip(columns, model.feature_importance("gain")), key=lambda p: -p[1])[:30]
        results[name] = {"holdout_auc": auc, "best_iteration": model.best_iteration,
                         "features": len(columns), "seconds": time.time() - began,
                         "top_gain": [[c, float(g)] for c, g in top]}
        print(name, json.dumps({k: v for k, v in results[name].items() if k != "top_gain"}), flush=True)
        (args.output / "results.json").write_text(json.dumps({
            "cutoff": CUTOFF.isoformat(), "valid_rows": int(len(y_valid)), "fit_rows": int(len(y_fit)),
            "early_stopping_on_holdout": True, "wide_features": wide, "results": results,
            "seconds": time.time() - start}, indent=2) + "\n")
        del train_set, valid_set, x_valid, model
        gc.collect()


if __name__ == "__main__":
    main()
