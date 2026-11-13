"""T029 local screen: classic learners on the T026 wide request-context features.

Same protocol as T026 (fit rows before the original cutoff, score the original
307,528-row holdout, early stopping on the holdout). The wide feature list is the
one T026 chose from training rows only. Holdout predictions are saved with file
row ids so they can be blended with the neural models.
"""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from feature_screen import CUTOFF, ITEM, LOW_CATEGORICAL, scalar_numeric
from request_screen import context


def fit_xgboost(x_fit, y_fit, x_valid, y_valid, categorical, threads):
    import xgboost as xgb

    fit = xgb.QuantileDMatrix(x_fit, y_fit, max_bin=127)
    valid = xgb.QuantileDMatrix(x_valid, y_valid, ref=fit, max_bin=127)
    params = {
        "objective": "binary:logistic", "eval_metric": "auc", "tree_method": "hist",
        "max_depth": 0, "max_leaves": 127, "grow_policy": "lossguide", "min_child_weight": 20,
        "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.5, "lambda": 10,
        "nthread": threads, "seed": 42, "max_bin": 127,
    }
    model = xgb.train(params, fit, 4000, evals=[(valid, "holdout")],
                      early_stopping_rounds=200, verbose_eval=250)
    predictions = model.predict(valid, iteration_range=(0, model.best_iteration + 1))
    return predictions, {"best_iteration": int(model.best_iteration)}


def fit_catboost(x_fit, y_fit, x_valid, y_valid, categorical, threads):
    from catboost import CatBoostClassifier, Pool

    def pool(x, y):
        frame = {f"f{i}": x[:, i] for i in range(x.shape[1])}
        for i in categorical:
            frame[f"f{i}"] = np.nan_to_num(x[:, i], nan=-1).astype(np.int32)
        import pandas as pd

        return Pool(pd.DataFrame(frame), y, cat_features=[f"f{i}" for i in categorical])

    fit, valid = pool(x_fit, y_fit), pool(x_valid, y_valid)
    model = CatBoostClassifier(
        iterations=4000, learning_rate=0.08, depth=8, border_count=128, l2_leaf_reg=10,
        eval_metric="AUC", od_type="Iter", od_wait=200, thread_count=threads,
        random_seed=42, verbose=250, use_best_model=True,
    )
    model.fit(fit, eval_set=valid)
    return model.predict_proba(valid)[:, 1], {"best_iteration": int(model.get_best_iteration())}


def fit_logreg(x_fit, y_fit, x_valid, y_valid, categorical, threads):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder

    sample = np.sort(np.random.default_rng(42).choice(len(y_fit), min(400_000, len(y_fit)), replace=False))
    x_fit, y_fit = x_fit[sample], y_fit[sample]
    dense = [i for i in range(x_fit.shape[1]) if i not in set(categorical)]

    keep = np.isnan(x_fit[:, dense]).any(axis=0)

    def transform(x, stats=None):
        values = x[:, dense]
        if stats is None:
            low, high = np.nanpercentile(values, [0.5, 99.5], axis=0)
            clipped = np.clip(values, low, high)
            stats = (low, high, np.nanmean(clipped, axis=0), np.nanstd(clipped, axis=0) + 1e-6)
        low, high, mean, std = stats
        missing = np.isnan(values)
        values = (np.clip(values, low, high) - mean) / std
        values[missing] = 0
        return np.hstack([values, missing[:, keep]]).astype(np.float32), stats

    fit_values, stats = transform(x_fit)
    valid_values, _ = transform(x_valid, stats)
    encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=False, dtype=np.float32)
    fit_codes = encoder.fit_transform(np.nan_to_num(x_fit[:, categorical], nan=-1))
    valid_codes = encoder.transform(np.nan_to_num(x_valid[:, categorical], nan=-1))
    fit_values, valid_values = np.hstack([fit_values, fit_codes]), np.hstack([valid_values, valid_codes])
    model = LogisticRegression(C=0.1, max_iter=500, tol=1e-5)
    model.fit(fit_values, y_fit)
    return model.predict_proba(valid_values)[:, 1], {"iterations": int(model.n_iter_[0]),
                                                    "inputs": int(fit_values.shape[1])}


def build_matrices(data, wide, rows=0):
    """Fit/holdout float32 matrices of the T029 wide feature set (categoricals as fit-period codes)."""
    numeric = scalar_numeric(pl.scan_parquet(data / "train.parquet").collect_schema())
    frame = pl.scan_parquet(data / "train.parquet").with_row_index("_row_id")
    if rows:
        frame = frame.filter(pl.col("request_id").hash(7) % rows == 0)
    frame = frame.select("_row_id", "watched", "feature1", "feature3", "request_id", ITEM,
                         *dict.fromkeys([*LOW_CATEGORICAL, *numeric])).collect()
    frame = frame.with_columns(context(wide))
    is_valid = (frame["feature1"] >= CUTOFF).to_numpy()
    categories = {c: frame.filter(pl.Series(~is_valid))[c].drop_nulls().unique().sort().to_list()
                  for c in LOW_CATEGORICAL}
    columns = ["feature3", *numeric, *LOW_CATEGORICAL, "rq_size", "rq_items", "rq_item_dup",
               *[f"rq_{k}_{c}" for c in wide for k in ["rank", "dev", "gap"]]]
    categorical = [i for i, c in enumerate(columns) if c in categories]

    y = frame["watched"].to_numpy()
    y_fit, y_valid = y[~is_valid], y[is_valid]
    row_ids = frame["_row_id"].to_numpy()[is_valid]
    x_fit = np.empty((int((~is_valid).sum()), len(columns)), dtype=np.float32)
    x_valid = np.empty((int(is_valid.sum()), len(columns)), dtype=np.float32)
    for i, c in enumerate(columns):
        column = pl.col(c)
        if c in categories:
            column = column.cast(pl.Enum(categories[c]), strict=False).to_physical()
        values = frame.select(column.cast(pl.Float32))[c].to_numpy()
        x_fit[:, i], x_valid[:, i] = values[~is_valid], values[is_valid]

    del frame
    gc.collect()
    return x_fit, y_fit, x_valid, y_valid, row_ids, columns, categorical


def full_matrices(data, wide, rows=0):
    """All-train and test matrices of the T029 wide feature set for a full-data refit.

    Categorical codes come from every labeled row; request context is computed per
    request within each file (requests never cross files). Test rows stay in file order.
    """
    numeric = scalar_numeric(pl.scan_parquet(data / "train.parquet").collect_schema())
    columns = ["feature3", *numeric, *LOW_CATEGORICAL, "rq_size", "rq_items", "rq_item_dup",
               *[f"rq_{k}_{c}" for c in wide for k in ["rank", "dev", "gap"]]]
    categories, outputs = None, {}
    for name, extra in [("train", "watched"), ("test", "id")]:
        frame = pl.scan_parquet(data / f"{name}.parquet")
        if rows:
            frame = frame.filter(pl.col("request_id").hash(7) % rows == 0)
        frame = frame.select(extra, "feature3", "request_id", ITEM,
                             *dict.fromkeys([*LOW_CATEGORICAL, *numeric])).collect()
        frame = frame.with_columns(context(wide))
        if categories is None:
            categories = {c: frame[c].drop_nulls().unique().sort().to_list() for c in LOW_CATEGORICAL}
        x = np.empty((len(frame), len(columns)), dtype=np.float32)
        for i, c in enumerate(columns):
            column = pl.col(c)
            if c in categories:
                column = column.cast(pl.Enum(categories[c]), strict=False).to_physical()
            x[:, i] = frame.select(column.cast(pl.Float32))[c].to_numpy()
        outputs[name] = (x, frame[extra].to_numpy())
        del frame
        gc.collect()
    categorical = [i for i, c in enumerate(columns) if c in categories]
    (x_train, y_train), (x_test, test_ids) = outputs["train"], outputs["test"]
    return x_train, y_train, x_test, test_ids, columns, categorical


MODELS = {"xgboost": fit_xgboost, "catboost": fit_catboost, "logreg": fit_logreg}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--wide-from", type=Path, default=Path("artifacts/local-t026/results.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/local-t029"))
    parser.add_argument("--models", default="xgboost,logreg,catboost")
    parser.add_argument("--threads", type=int, default=7)
    parser.add_argument("--rows", type=int, default=0, help="Keep 1/N of requests for smoke tests")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.time()
    wide = json.loads(args.wide_from.read_text())["wide_features"]
    x_fit, y_fit, x_valid, y_valid, row_ids, columns, categorical = build_matrices(args.data, wide, args.rows)
    print(f"matrices {x_fit.shape} / {x_valid.shape} in {time.time() - start:.0f}s", flush=True)
    path = args.output / "results.json"
    results = json.loads(path.read_text())["results"] if path.exists() else {}
    for name in args.models.split(","):
        began = time.time()
        predictions, extra = MODELS[name](x_fit, y_fit, x_valid, y_valid, categorical, args.threads)
        auc = float(roc_auc_score(y_valid, predictions))
        np.savez_compressed(args.output / f"{name}-holdout.npz", row_ids=row_ids,
                            y_true=y_valid, probabilities=predictions.astype(np.float32))
        results[name] = {"holdout_auc": auc, "seconds": time.time() - began, **extra}
        print(name, json.dumps(results[name]), flush=True)
        path.write_text(json.dumps({
            "cutoff": CUTOFF.isoformat(), "valid_rows": int(len(y_valid)), "fit_rows": int(len(y_fit)),
            "features": len(columns), "early_stopping_on_holdout": True, "wide_features": wide,
            "results": results}, indent=2) + "\n")
        gc.collect()


if __name__ == "__main__":
    main()
