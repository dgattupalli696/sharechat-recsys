"""Local LightGBM screen of request-context, train+test count and target-encoding features.

Trains on rows before the original primary cutoff and scores the original
307,528-row holdout. request_id is used only for grouping, never as an input.
"""
import argparse
from datetime import datetime
import gc
import json
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

CUTOFF = datetime.fromisoformat("2026-10-28T08:06:52.397000")
USER, ITEM = "feature5", "feature18"
LOW_CATEGORICAL = [f"feature{i}" for i in [2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 22, 25]]
COUNT_KEYS = [USER, ITEM, "feature23", "feature20", "feature19", "feature9", "feature12"]
ENCODED = [ITEM, "feature23", "feature20", "feature19", "feature9", "feature12", "feature21", USER]
RANKED = ["feature418", "feature419", "feature411", "feature410", "feature430",
          "feature431", "feature417", "feature352", "feature346", "feature480"]
SMOOTHING = 20.0


def scalar_numeric(schema):
    skip = {"feature1", "feature3"}
    return [c for c, t in schema.items()
            if c.startswith("feature") and c not in skip
            and str(t) in {"Float32", "Int64", "Int8"}]


def time_seconds(column):
    return pl.col(column).dt.epoch("ms").cast(pl.Float64) / 1000.0


def count_features(keys):
    """Label-free counts over the complete train+test population."""
    frame = keys.with_columns(time_seconds("feature1").alias("t"))
    exprs = []
    for key in COUNT_KEYS:
        exprs += [pl.len().over(key).alias(f"cnt_{key}"),
                  pl.col("request_id").n_unique().over(key).alias(f"req_{key}")]
    frame = frame.with_columns(exprs)
    pair = [USER, ITEM]
    frame = frame.with_columns(
        pl.len().over(pair).alias("cnt_user_item"),
        pl.col("t").min().over(ITEM).alias("item_first"),
        pl.col("t").max().over(ITEM).alias("item_last"),
        pl.col("t").min().over(USER).alias("user_first"),
        pl.col("t").max().over(USER).alias("user_last"),
        pl.len().over([ITEM, pl.col("feature1").dt.truncate("1h")]).alias("cnt_item_hour"),
        pl.len().over([ITEM, pl.col("feature1").dt.truncate("1d")]).alias("cnt_item_day"),
    )
    requests = (
        frame.group_by("request_id").agg(pl.col(USER).first(), pl.col("t").first())
        .sort([USER, "t", "request_id"])
        .with_columns(
            pl.int_range(pl.len()).over(USER).alias("user_req_index"),
            (pl.col("t") - pl.col("t").shift(1).over(USER)).alias("user_prev_gap"),
            (pl.col("t").shift(-1).over(USER) - pl.col("t")).alias("user_next_gap"),
        )
        .select("request_id", "user_req_index", "user_prev_gap", "user_next_gap")
    )
    frame = frame.join(requests, on="request_id", how="left")
    pairs = (
        frame.select(USER, ITEM, "t").unique().sort([USER, ITEM, "t"])
        .with_columns(
            pl.int_range(pl.len()).over([USER, ITEM]).alias("user_item_index"),
            (pl.col("t") - pl.col("t").shift(1).over([USER, ITEM])).alias("user_item_prev_gap"),
            (pl.col("t").shift(-1).over([USER, ITEM]) - pl.col("t")).alias("user_item_next_gap"),
        )
    )
    frame = frame.join(pairs, on=[USER, ITEM, "t"], how="left")
    frame = frame.with_columns(
        (pl.col("req_feature5") - pl.col("user_req_index") - 1).alias("user_reqs_after"),
        (pl.col("t") - pl.col("item_first")).alias("item_age"),
        (pl.col("item_last") - pl.col("t")).alias("item_remaining"),
        (pl.col("t") - pl.col("user_first")).alias("user_since_first"),
        (pl.col("user_last") - pl.col("t")).alias("user_until_last"),
    ).drop("item_first", "item_last", "user_first", "user_last")
    names = [c for c in frame.columns if c not in keys.columns and c != "t"]
    return frame.select("row", *names), names


def request_features(frame):
    """Within-request context; frame is one split population with numeric columns."""
    group = "request_id"
    exprs = [pl.len().over(group).alias("rq_size"),
             pl.col(ITEM).n_unique().over(group).alias("rq_items"),
             pl.len().over([group, ITEM]).alias("rq_item_dup")]
    for c in RANKED:
        exprs += [
            (pl.col(c).rank("average").over(group) / pl.len().over(group)).alias(f"rq_rank_{c}"),
            (pl.col(c) - pl.col(c).mean().over(group)).alias(f"rq_dev_{c}"),
            (pl.col(c) - pl.col(c).max().over(group)).alias(f"rq_gap_{c}"),
        ]
    out = frame.select("row", *exprs)
    return out, [c for c in out.columns if c != "row"]


def target_encode(train, valid, column, folds):
    prior = float(train["watched"].mean())
    y = train["watched"].to_numpy().astype(np.float64)
    keys = train[column]
    oof = np.empty(len(train))
    support = np.empty(len(train))
    for fit, apply in folds:
        stats = (pl.DataFrame({"k": keys[fit], "y": y[fit]})
                 .group_by("k").agg(pl.col("y").sum().alias("s"), pl.len().alias("n")))
        joined = pl.DataFrame({"k": keys[apply]}).join(stats, on="k", how="left").fill_null(0)
        s, n = joined["s"].to_numpy(), joined["n"].to_numpy()
        oof[apply] = (s + SMOOTHING * prior) / (n + SMOOTHING)
        support[apply] = n
    stats = (pl.DataFrame({"k": keys, "y": y})
             .group_by("k").agg(pl.col("y").sum().alias("s"), pl.len().alias("n")))
    joined = pl.DataFrame({"k": valid[column]}).join(stats, on="k", how="left").fill_null(0)
    s, n = joined["s"].to_numpy(), joined["n"].to_numpy()
    held = (s + SMOOTHING * prior) / (n + SMOOTHING)
    return (oof, support), (held, n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/local-t023"))
    parser.add_argument("--variants", default="base,request,counts,target,all")
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--rounds", type=int, default=4000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--rows", type=int, default=0, help="Keep 1/N of requests for smoke tests")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.time()
    schema = pl.scan_parquet(args.data / "train.parquet").collect_schema()
    numeric = scalar_numeric(schema)
    key_columns = ["request_id", "feature1", *sorted(set(COUNT_KEYS + ENCODED + LOW_CATEGORICAL), key=lambda c: int(c[7:]))]
    train = pl.scan_parquet(args.data / "train.parquet").with_row_index("row")
    if args.rows:
        train = train.filter(pl.col("request_id").hash(7) % args.rows == 0)
    train = train.select("row", "watched", "feature3", *key_columns, *numeric).collect()
    test_keys = pl.scan_parquet(args.data / "test.parquet").select(*key_columns)
    if args.rows:
        test_keys = test_keys.filter(pl.col("request_id").hash(7) % args.rows == 0)
    test_keys = test_keys.collect()
    offset = int(train["row"].max()) + 1
    keys = pl.concat([
        train.select("row", *key_columns),
        test_keys.with_row_index("row", offset=offset).with_columns(pl.col("row").cast(train["row"].dtype)),
    ])
    del test_keys
    counts, count_names = count_features(keys)
    del keys
    gc.collect()
    train = train.join(counts, on="row", how="left")
    del counts
    rq, rq_names = request_features(train.select("row", "request_id", ITEM, *RANKED))
    train = train.join(rq, on="row", how="left")
    del rq
    is_valid = (train["feature1"] >= CUTOFF).to_numpy()
    fit_rows, valid_rows = train.filter(~pl.Series(is_valid)), train.filter(pl.Series(is_valid))
    del train
    gc.collect()
    folds = list(GroupKFold(5).split(np.zeros(len(fit_rows)), groups=fit_rows["request_id"].to_numpy()))
    te_fit, te_valid, te_names = {}, {}, []
    for column in ENCODED:
        (a, an), (b, bn) = target_encode(fit_rows, valid_rows, column, folds)
        te_fit[f"te_{column}"], te_valid[f"te_{column}"] = a, b
        te_fit[f"ten_{column}"], te_valid[f"ten_{column}"] = an, bn
        te_names += [f"te_{column}", f"ten_{column}"]
    categories = {}
    for c in LOW_CATEGORICAL:
        vocabulary = fit_rows[c].drop_nulls().unique().sort()
        categories[c] = vocabulary
    base = ["feature3", *numeric, *LOW_CATEGORICAL]
    groups = {"base": base, "request": base + rq_names, "counts": base + count_names,
              "target": base + te_names, "all": base + rq_names + count_names + te_names}

    def matrix(frame, te, columns):
        parts = []
        for c in columns:
            if c in te:
                parts.append(te[c].astype(np.float32))
            elif c in categories:
                codes = frame.select(pl.col(c).cast(pl.Enum(categories[c].to_list()), strict=False)
                                     .to_physical().cast(pl.Float32))[c].to_numpy()
                parts.append(codes)
            else:
                parts.append(frame[c].cast(pl.Float32).to_numpy())
        return np.ascontiguousarray(np.column_stack(parts), dtype=np.float32)

    y_fit = fit_rows["watched"].to_numpy()
    y_valid = valid_rows["watched"].to_numpy()
    valid_ids = valid_rows["row"].to_numpy()
    results = {}
    for name in args.variants.split(","):
        columns = groups[name]
        began = time.time()
        x_fit = matrix(fit_rows, te_fit, columns)
        x_valid = matrix(valid_rows, te_valid, columns)
        categorical = [i for i, c in enumerate(columns) if c in categories]
        train_set = lgb.Dataset(x_fit, y_fit, feature_name=columns,
                                categorical_feature=categorical, free_raw_data=True)
        valid_set = lgb.Dataset(x_valid, y_valid, reference=train_set)
        params = {
            "objective": "binary", "metric": "auc", "learning_rate": args.learning_rate,
            "num_leaves": 127, "min_data_in_leaf": 200, "feature_fraction": 0.5,
            "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10,
            "max_bin": 127, "seed": 42, "num_threads": args.threads, "verbosity": -1,
            "cat_smooth": 20, "max_cat_to_onehot": 8,
        }
        log = {}
        model = lgb.train(params, train_set, args.rounds, valid_sets=[valid_set], valid_names=["holdout"],
                          callbacks=[lgb.early_stopping(200, verbose=False), lgb.record_evaluation(log),
                                     lgb.log_evaluation(250)])
        prediction = model.predict(x_valid, num_iteration=model.best_iteration)
        auc = float(roc_auc_score(y_valid, prediction))
        curve = log["holdout"]["auc"]
        importance = sorted(zip(columns, model.feature_importance("gain")), key=lambda p: -p[1])[:30]
        results[name] = {
            "holdout_auc": auc, "best_iteration": model.best_iteration,
            "auc_at_140": curve[min(139, len(curve) - 1)], "features": len(columns),
            "seconds": time.time() - began,
            "top_gain": [[c, float(g)] for c, g in importance],
        }
        np.save(args.output / f"valid-{name}.npy", prediction.astype(np.float32))
        print(name, json.dumps({k: v for k, v in results[name].items() if k != "top_gain"}), flush=True)
        print(" top:", [c for c, _ in importance[:15]], flush=True)
        (args.output / "results.json").write_text(json.dumps({
            "cutoff": CUTOFF.isoformat(), "valid_rows": int(len(y_valid)),
            "fit_rows": int(len(y_fit)), "early_stopping_on_holdout": True,
            "results": results, "seconds": time.time() - start}, indent=2) + "\n")
        del x_fit, x_valid, train_set, valid_set, model
        gc.collect()
    np.save(args.output / "valid-row-ids.npy", valid_ids)


if __name__ == "__main__":
    main()
