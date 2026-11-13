"""T032 local LightGBM screen of a two-stage (hurdle) watch model.

P(watched) = P(request has any watch) x P(row watched | request has a watch).
Stage 1 is trained on one row per request (request-constant columns plus feed
summaries); stage 2 only on fit requests with at least one watch. Both stages and the
T031 single-model control are scored on the original 307,528-row holdout, then combined
by product and by a request-grouped 2-fold logistic stacker fitted on the holdout.
"""
import argparse
import gc
import json
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from cf_screen import within_request_auc
from feature_screen import CUTOFF, ITEM, LOW_CATEGORICAL, RANKED, request_features, scalar_numeric

PARAMS = {
    "objective": "binary", "metric": "auc", "num_leaves": 127, "min_data_in_leaf": 200,
    "feature_fraction": 0.5, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10,
    "max_bin": 127, "seed": 42, "verbosity": -1, "cat_smooth": 20, "max_cat_to_onehot": 8,
}


def fit(x_fit, y_fit, x_valid, y_valid, names, categorical, args):
    params = {**PARAMS, "learning_rate": args.learning_rate, "num_threads": args.threads}
    train_set = lgb.Dataset(x_fit, y_fit, feature_name=names, categorical_feature=categorical, free_raw_data=True)
    valid_set = lgb.Dataset(x_valid, y_valid, reference=train_set)
    model = lgb.train(params, train_set, args.rounds, valid_sets=[valid_set], valid_names=["holdout"],
                      callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(250)])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--control", type=Path, default=Path("artifacts/local-t031"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/local-t032"))
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--rounds", type=int, default=4000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--rows", type=int, default=0, help="Keep 1/N of requests for smoke tests")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.time()
    numeric = scalar_numeric(pl.scan_parquet(args.data / "train.parquet").collect_schema())
    keys = sorted(set([ITEM] + LOW_CATEGORICAL), key=lambda c: int(c[7:]))
    lf = pl.scan_parquet(args.data / "train.parquet").with_row_index("row")
    if args.rows:
        lf = lf.filter(pl.col("request_id").hash(7) % args.rows == 0)
    train = lf.select("row", "watched", "request_id", "feature1", "feature3", *keys,
                      *[c for c in numeric if c not in keys]).collect()
    rq, rq_names = request_features(train.select("row", "request_id", ITEM, *RANKED))
    train = train.join(rq, on="row", how="left").with_columns(
        (pl.col("watched").max().over("request_id")).alias("any_watch"))
    del rq
    categories = {c: train.filter(pl.col("feature1") < CUTOFF)[c].drop_nulls().unique().sort() for c in LOW_CATEGORICAL}
    row_columns = ["feature3", *[c for c in numeric if c not in categories], *LOW_CATEGORICAL, *rq_names]
    sample = train.head(200_000)
    counts = sample.group_by("request_id").agg([pl.col(c).n_unique() for c in row_columns])
    constant = [c for c in row_columns if float((counts[c] <= 1).mean()) > 0.99]
    varying = [c for c in RANKED + [c for c in rq_names if not c.startswith("rq_rank")] if c not in constant][:40]
    del sample, counts

    def encode(frame, columns):
        parts = []
        for c in columns:
            if c in categories:
                parts.append(frame.select(pl.col(c).cast(pl.Enum(categories[c].to_list()), strict=False)
                                          .to_physical().cast(pl.Float32))[c].to_numpy())
            else:
                parts.append(frame[c].cast(pl.Float32).to_numpy())
        return np.ascontiguousarray(np.column_stack(parts), dtype=np.float32)

    request_level = train.group_by("request_id", maintain_order=True).agg(
        pl.col("feature1").first(), pl.col("any_watch").first(),
        *[pl.col(c).first() for c in constant],
        *[pl.col(c).cast(pl.Float32).max().alias(f"max_{c}") for c in varying],
        *[pl.col(c).cast(pl.Float32).mean().alias(f"mean_{c}") for c in varying],
        *[pl.col(c).cast(pl.Float32).std().alias(f"std_{c}") for c in varying])
    request_columns = [c for c in request_level.columns if c not in {"request_id", "feature1", "any_watch"}]
    held = request_level["feature1"] >= CUTOFF
    fit_req, valid_req = request_level.filter(~held), request_level.filter(held)
    del request_level
    names = request_columns
    categorical = [i for i, c in enumerate(names) if c in categories]
    began = time.time()
    stage1 = fit(encode(fit_req, names), fit_req["any_watch"].to_numpy(), encode(valid_req, names),
                 valid_req["any_watch"].to_numpy(), names, categorical, args)
    p1 = stage1.predict(encode(valid_req, names), num_iteration=stage1.best_iteration)
    results = {"stage1": {"request_auc": float(roc_auc_score(valid_req["any_watch"], p1)),
                          "best_iteration": stage1.best_iteration, "features": len(names),
                          "requests_fit": len(fit_req), "seconds": time.time() - began}}
    print("stage1", json.dumps(results["stage1"]), flush=True)
    stage1_frame = pl.DataFrame({"request_id": valid_req["request_id"], "p1": p1})
    del stage1, fit_req, valid_req
    gc.collect()

    is_valid = (train["feature1"] >= CUTOFF).to_numpy()
    fit_rows = train.filter(~pl.Series(is_valid) & (pl.col("any_watch") == 1))
    valid_rows = train.filter(pl.Series(is_valid))
    del train
    gc.collect()
    positive = valid_rows["any_watch"].to_numpy() == 1
    x_valid = encode(valid_rows, row_columns)
    categorical = [i for i, c in enumerate(row_columns) if c in categories]
    began = time.time()
    stage2 = fit(encode(fit_rows, row_columns), fit_rows["watched"].to_numpy(), x_valid[positive],
                 valid_rows["watched"].to_numpy()[positive], row_columns, categorical, args)
    p2 = stage2.predict(x_valid, num_iteration=stage2.best_iteration)
    y = valid_rows["watched"].to_numpy()
    results["stage2"] = {"positive_request_auc": float(roc_auc_score(y[positive], p2[positive])),
                         "within_request_auc": within_request_auc(valid_rows["request_id"], y, p2),
                         "best_iteration": stage2.best_iteration, "fit_rows": len(fit_rows),
                         "seconds": time.time() - began,
                         "top_gain": [c for c, _ in sorted(zip(row_columns, stage2.feature_importance("gain")),
                                                           key=lambda p: -p[1])[:15]]}
    print("stage2", json.dumps(results["stage2"]), flush=True)
    del stage2, fit_rows, x_valid
    gc.collect()

    frame = valid_rows.select("row", "request_id", "watched").with_columns(pl.Series("p2", p2)).join(
        stage1_frame, on="request_id", how="left")
    control_ids = np.load(args.control / "valid-row-ids.npy")
    control = pl.DataFrame({"row": control_ids, "single": np.load(args.control / "valid-control.npy")})
    frame = frame.join(control.with_columns(pl.col("row").cast(frame["row"].dtype)), on="row", how="left")
    requests = frame["request_id"]
    y = frame["watched"].to_numpy()
    eps = 1e-6
    l1, l2 = logit(np.clip(frame["p1"].to_numpy(), eps, 1 - eps)), logit(np.clip(frame["p2"].to_numpy(), eps, 1 - eps))
    single = frame["single"].to_numpy()
    product = frame["p1"].to_numpy() * frame["p2"].to_numpy()

    def report(name, prediction):
        results[name] = {"holdout_auc": float(roc_auc_score(y, prediction)),
                         "within_request_auc": within_request_auc(requests, y, prediction)}
        print(name, json.dumps(results[name]), flush=True)

    report("single", single)
    report("product", product)
    for name, features in [("stack_stages", [l1, l2]),
                           ("stack_stages_single", [l1, l2, logit(np.clip(single, eps, 1 - eps))])]:
        x = np.column_stack(features)
        stacked = np.empty(len(y))
        for a, b in GroupKFold(2).split(x, y, groups=requests.to_numpy()):
            stacked[b] = LogisticRegression(C=1.0, max_iter=1000).fit(x[a], y[a]).decision_function(x[b])
        report(name, stacked)
    results.update({"cutoff": CUTOFF.isoformat(), "valid_rows": int(len(y)), "early_stopping_on_holdout": True,
                    "stacker": "2-fold request-grouped logistic regression on holdout", "seconds": time.time() - start})
    (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    np.savez(args.output / "holdout.npz", row_ids=frame["row"].to_numpy(), p1=frame["p1"].to_numpy(), p2=p2)


if __name__ == "__main__":
    main()
