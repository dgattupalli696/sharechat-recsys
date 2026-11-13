"""T033 local screen: residual boosting / stacking XGBoost on top of the T027 TabM.

Works only on the original 307,528-row holdout, which has out-of-sample predictions from
every model. Two request-grouped folds: XGBoost is trained on one half (10% of its
requests used for early stopping) and scores the other half. Variants: (a) init_score =
T027 logit (residual), (b) T027 logit as a feature, (c) all model logits as features.
References: T027 alone and the 0.7 T027 + 0.3 T029-XGBoost rank blend.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import polars as pl
import xgboost as xgb
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from cf_screen import within_request_auc
from feature_screen import CUTOFF, ITEM, LOW_CATEGORICAL, scalar_numeric
from request_screen import context

PREDICTIONS = {
    "t027": "artifacts/kaggle-t027/t027/ensemble-32/validation_predictions.npz",
    "t028": "artifacts/kaggle-t028/t028/validation_predictions.npz",
    "k32": "artifacts/kaggle-t024/t024/ensemble-32/validation_predictions.npz",
    "dcn": "artifacts/kaggle-t024/t024/dcnv2-r32/validation_predictions.npz",
    "plr": "artifacts/kaggle-t024/t024/cross-masked-plr/validation_predictions.npz",
    "lgb": "artifacts/local-t024-lightgbm/lightgbm/validation_predictions.npz",
    "xgb": "artifacts/kaggle-t029/t029/xgboost-holdout.npz",
}


def holdout_matrix(data, wide):
    numeric = scalar_numeric(pl.scan_parquet(data / "train.parquet").collect_schema())
    scan = pl.scan_parquet(data / "train.parquet").with_row_index("_row_id")
    categories = {c: scan.filter(pl.col("feature1") < CUTOFF).select(pl.col(c).drop_nulls().unique().sort())
                  .collect()[c].to_list() for c in LOW_CATEGORICAL}
    frame = (scan.filter(pl.col("feature1") >= CUTOFF)
             .select("_row_id", "watched", "feature3", "request_id", ITEM, *dict.fromkeys([*LOW_CATEGORICAL, *numeric]))
             .collect().with_columns(context(wide)))
    columns = ["feature3", *numeric, *LOW_CATEGORICAL, "rq_size", "rq_items", "rq_item_dup",
               *[f"rq_{k}_{c}" for c in wide for k in ["rank", "dev", "gap"]]]
    x = np.empty((len(frame), len(columns)), dtype=np.float32)
    for i, c in enumerate(columns):
        column = pl.col(c)
        if c in categories:
            column = column.cast(pl.Enum(categories[c]), strict=False).to_physical()
        x[:, i] = frame.select(column.cast(pl.Float32))[c].to_numpy()
    return frame.select("_row_id", "watched", "request_id"), x, columns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--wide-from", type=Path, default=Path("artifacts/local-t026/results.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/local-t033"))
    parser.add_argument("--threads", type=int, default=7)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.time()
    keys, x, columns = holdout_matrix(args.data, json.loads(args.wide_from.read_text())["wide_features"])
    keys = keys.with_columns(pl.col("_row_id").cast(pl.Int64))
    for name, path in PREDICTIONS.items():
        z = np.load(path)
        keys = keys.join(
            pl.DataFrame({"_row_id": z["row_ids"].astype(np.int64), name: z["probabilities"].astype(np.float64)}),
            on="_row_id", how="left", maintain_order="left")
    assert keys.select(pl.all().null_count()).sum_horizontal().item() == 0
    print(f"holdout matrix {x.shape} in {time.time() - start:.0f}s", flush=True)
    y = keys["watched"].to_numpy()
    requests = keys["request_id"].to_numpy()
    logits = {m: logit(np.clip(keys[m].to_numpy(), 1e-6, 1 - 1e-6)) for m in PREDICTIONS}
    t027 = logits["t027"]
    variants = {
        "residual_t027": (x, t027),
        "feature_t027": (np.column_stack([x, t027]).astype(np.float32), None),
        "stack_all": (np.column_stack([x, *logits.values()]).astype(np.float32), None),
    }
    params = {
        "objective": "binary:logistic", "eval_metric": "auc", "tree_method": "hist",
        "max_depth": 0, "max_leaves": 31, "grow_policy": "lossguide", "min_child_weight": 50,
        "eta": 0.03, "subsample": 0.8, "colsample_bytree": 0.5, "lambda": 10,
        "nthread": args.threads, "seed": 42, "max_bin": 127,
    }
    out = {name: np.empty(len(y)) for name in variants}
    iterations = {name: [] for name in variants}
    folds = list(GroupKFold(2).split(x, y, groups=requests))
    for fold, (train, test) in enumerate(folds):
        inner = (pl.Series(requests[train]).hash(13) % 10 == 0).to_numpy()
        fit, stop = train[~inner], train[inner]
        for name, (matrix, base) in variants.items():
            began = time.time()
            margin = (lambda idx: base[idx]) if base is not None else (lambda idx: None)
            d_fit = xgb.QuantileDMatrix(matrix[fit], y[fit], base_margin=margin(fit), max_bin=127)
            d_stop = xgb.QuantileDMatrix(matrix[stop], y[stop], base_margin=margin(stop), ref=d_fit, max_bin=127)
            d_test = xgb.QuantileDMatrix(matrix[test], base_margin=margin(test), ref=d_fit, max_bin=127)
            model = xgb.train(params, d_fit, 3000, evals=[(d_stop, "stop")], early_stopping_rounds=200, verbose_eval=False)
            out[name][test] = model.predict(d_test, iteration_range=(0, model.best_iteration + 1), output_margin=True)
            iterations[name].append(int(model.best_iteration))
            print(f"fold {fold} {name}: best_iteration {model.best_iteration}, "
                  f"fold AUC {roc_auc_score(y[test], out[name][test]):.5f} ({time.time() - began:.0f}s)", flush=True)
    n = len(y)
    references = {"t027": t027, "blend_t027_xgb": 0.7 * rankdata(t027) / n + 0.3 * rankdata(logits["xgb"]) / n,
                  "blend_all_mean_rank": np.mean([rankdata(v) / n for v in logits.values()], axis=0)}
    results = {}
    halves = [test for _, test in folds]
    for name, score in {**references, **out}.items():
        results[name] = {"holdout_auc": float(roc_auc_score(y, score)),
                         "fold_auc": [float(roc_auc_score(y[h], score[h])) for h in halves],
                         "within_request_auc": within_request_auc(requests, y, score),
                         **({"best_iterations": iterations[name]} if name in iterations else {})}
        print(name, json.dumps(results[name]), flush=True)
    for name in out:
        for w in [0.3, 0.5, 0.7]:
            score = w * rankdata(out[name]) / n + (1 - w) * references["blend_t027_xgb"]
            results[f"{name}+blend@{w}"] = {"holdout_auc": float(roc_auc_score(y, score)),
                                            "fold_auc": [float(roc_auc_score(y[h], score[h])) for h in halves]}
            print(f"{name}+blend@{w}", json.dumps(results[f"{name}+blend@{w}"]), flush=True)
    (args.output / "results.json").write_text(json.dumps({
        "protocol": "2-fold request-grouped cross-fit on the original holdout; 10% inner requests for early stopping",
        "features": len(columns), "results": results, "seconds": time.time() - start}, indent=2) + "\n")
    np.savez(args.output / "holdout.npz", row_ids=keys["_row_id"].to_numpy(), y_true=y, **out)


if __name__ == "__main__":
    main()
