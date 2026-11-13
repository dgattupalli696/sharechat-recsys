"""T031 local LightGBM screen of user-cluster collaborative-filtering target encodings.

Users are almost always new, so "similar users" are defined by clustering request-level
profiles (request-constant scalar columns, dates excluded) and the feature32 user vector
over train+test requests without labels. Out-of-fold (cluster x item/creator/category)
watch rates are then added to the T023 request-context control. Scores the original
307,528-row holdout; request_id is only used for grouping.
"""
import argparse
import gc
import json
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from feature_screen import CUTOFF, ITEM, LOW_CATEGORICAL, RANKED, request_features, scalar_numeric, target_encode

DATES = {"feature1", "feature2", "feature3", "feature4"}
USER_VECTOR = "feature32"
ITEM_KEYS = [ITEM, "feature23", "feature19"]
CROSSES = [("pc64", ITEM), ("pc512", ITEM), ("pc512", "feature23"), ("pc64", "feature19"),
           ("uv256", ITEM), ("uv256", "feature23")]


def request_constant(frame, columns):
    first = frame.filter(pl.col("request_id").is_first_distinct())
    counts = frame.group_by("request_id").agg([pl.col(c).n_unique() for c in columns])
    return [c for c in columns if float((counts[c] <= 1).mean()) > 0.99], first


def clusters(data, rows, seed=42):
    """Label-free request clusters over train+test."""
    schema = pl.scan_parquet(data / "train.parquet").collect_schema()
    numeric = [c for c in scalar_numeric(schema) if c not in DATES]

    def scan(name):
        lf = pl.scan_parquet(data / name)
        if rows:
            lf = lf.filter(pl.col("request_id").hash(7) % rows == 0)
        return lf
    probe = scan("train.parquet").select("request_id", *numeric).head(200_000).collect()
    profile, _ = request_constant(probe, numeric)
    del probe
    parts = [scan(n).filter(pl.col("request_id").is_first_distinct())
             .select("request_id", *profile, pl.col(USER_VECTOR)).collect() for n in ["train.parquet", "test.parquet"]]
    req = pl.concat(parts)
    del parts
    ranked = req.select([(pl.col(c).rank("average") / pl.col(c).count()).fill_null(0.5).fill_nan(0.5)
                         .cast(pl.Float32) for c in profile]).to_numpy()
    reduced = PCA(16, random_state=seed).fit_transform(ranked - ranked.mean(0))
    del ranked
    out = {"request_id": req["request_id"]}
    for k in [64, 512]:
        out[f"pc{k}"] = MiniBatchKMeans(k, batch_size=8192, n_init=3, random_state=seed).fit_predict(reduced)
    vectors = np.array([v if v is not None and len(v) == 32 else [0.0] * 32 for v in req[USER_VECTOR].to_list()],
                       dtype=np.float32)
    present = np.linalg.norm(vectors, axis=1) > 0
    unit = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-6)
    labels = np.full(len(unit), -1)
    labels[present] = MiniBatchKMeans(256, batch_size=8192, n_init=3, random_state=seed).fit_predict(unit[present])
    out["uv256"] = labels
    return pl.DataFrame(out).with_columns(pl.col("pc64", "pc512", "uv256").cast(pl.Int32)), len(profile)


def within_request_auc(requests, y, prediction):
    """AUC of rows in mixed-outcome requests after removing each request's mean prediction."""
    frame = pl.DataFrame({"r": requests, "y": y, "p": prediction}).with_columns(
        (pl.col("p") - pl.col("p").mean().over("r")).alias("d"),
        pl.col("y").sum().over("r").alias("s"), pl.len().over("r").alias("n"))
    mixed = frame.filter((pl.col("s") > 0) & (pl.col("s") < pl.col("n")))
    return float(roc_auc_score(mixed["y"], mixed["d"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/local-t031"))
    parser.add_argument("--variants", default="control,item_te,cf")
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--rounds", type=int, default=4000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--rows", type=int, default=0, help="Keep 1/N of requests for smoke tests")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.time()
    assignments, n_profile = clusters(args.data, args.rows)
    print(f"clusters ready ({n_profile} profile columns) {time.time() - start:.0f}s", flush=True)
    schema = pl.scan_parquet(args.data / "train.parquet").collect_schema()
    numeric = scalar_numeric(schema)
    train = pl.scan_parquet(args.data / "train.parquet").with_row_index("row")
    if args.rows:
        train = train.filter(pl.col("request_id").hash(7) % args.rows == 0)
    keys = sorted(set(ITEM_KEYS + LOW_CATEGORICAL), key=lambda c: int(c[7:]))
    train = (train.select("row", "watched", "request_id", "feature1", "feature3", *keys,
                          *[c for c in numeric if c not in keys]).collect()
             .join(assignments, on="request_id", how="left"))
    del assignments
    rq, rq_names = request_features(train.select("row", "request_id", ITEM, *RANKED))
    train = train.join(rq, on="row", how="left")
    del rq
    cross_names = []
    for a, b in CROSSES:
        name = f"x_{a}_{b}"
        train = train.with_columns(pl.struct(a, b).hash(11).alias(name))
        cross_names.append(name)
    is_valid = (train["feature1"] >= CUTOFF).to_numpy()
    fit_rows, valid_rows = train.filter(~pl.Series(is_valid)), train.filter(pl.Series(is_valid))
    del train
    gc.collect()
    folds = list(GroupKFold(5).split(np.zeros(len(fit_rows)), groups=fit_rows["request_id"].to_numpy()))
    te_fit, te_valid = {}, {}

    def encode(columns):
        names = []
        for column in columns:
            (a, an), (b, bn) = target_encode(fit_rows, valid_rows, column, folds)
            te_fit[f"te_{column}"], te_valid[f"te_{column}"] = a, b
            te_fit[f"ten_{column}"], te_valid[f"ten_{column}"] = np.log1p(an), np.log1p(bn)
            names += [f"te_{column}", f"ten_{column}"]
        return names
    item_te = encode(ITEM_KEYS)
    cf_te = encode(cross_names)
    for column, (a, b) in zip(cross_names, CROSSES):
        lift = f"lift_{column}"
        te_fit[lift] = te_fit[f"te_{column}"] - te_fit[f"te_{b}"]
        te_valid[lift] = te_valid[f"te_{column}"] - te_valid[f"te_{b}"]
        cf_te.append(lift)
    categories = {c: fit_rows[c].drop_nulls().unique().sort() for c in LOW_CATEGORICAL}
    base = ["feature3", *[c for c in numeric if c not in categories], *LOW_CATEGORICAL, *rq_names]
    groups = {"control": base, "item_te": base + item_te, "cf": base + item_te + cf_te}

    def matrix(frame, te, columns):
        parts = []
        for c in columns:
            if c in te:
                parts.append(te[c].astype(np.float32))
            elif c in categories:
                parts.append(frame.select(pl.col(c).cast(pl.Enum(categories[c].to_list()), strict=False)
                                          .to_physical().cast(pl.Float32))[c].to_numpy())
            else:
                parts.append(frame[c].cast(pl.Float32).to_numpy())
        return np.ascontiguousarray(np.column_stack(parts), dtype=np.float32)

    y_fit, y_valid = fit_rows["watched"].to_numpy(), valid_rows["watched"].to_numpy()
    univariate = {c: float(roc_auc_score(y_valid, te_valid[c])) for c in item_te + cf_te if c.startswith(("te_", "lift_"))}
    print("univariate holdout AUC:", json.dumps({k: round(v, 4) for k, v in univariate.items()}), flush=True)
    results = {}
    for name in args.variants.split(","):
        columns = groups[name]
        began = time.time()
        x_fit, x_valid = matrix(fit_rows, te_fit, columns), matrix(valid_rows, te_valid, columns)
        categorical = [i for i, c in enumerate(columns) if c in categories]
        train_set = lgb.Dataset(x_fit, y_fit, feature_name=columns, categorical_feature=categorical, free_raw_data=True)
        valid_set = lgb.Dataset(x_valid, y_valid, reference=train_set)
        params = {
            "objective": "binary", "metric": "auc", "learning_rate": args.learning_rate,
            "num_leaves": 127, "min_data_in_leaf": 200, "feature_fraction": 0.5,
            "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10,
            "max_bin": 127, "seed": 42, "num_threads": args.threads, "verbosity": -1,
            "cat_smooth": 20, "max_cat_to_onehot": 8,
        }
        model = lgb.train(params, train_set, args.rounds, valid_sets=[valid_set], valid_names=["holdout"],
                          callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(250)])
        prediction = model.predict(x_valid, num_iteration=model.best_iteration)
        importance = sorted(zip(columns, model.feature_importance("gain")), key=lambda p: -p[1])[:30]
        results[name] = {"holdout_auc": float(roc_auc_score(y_valid, prediction)),
                         "within_request_auc": within_request_auc(valid_rows["request_id"], y_valid, prediction),
                         "best_iteration": model.best_iteration, "features": len(columns),
                         "seconds": time.time() - began, "top_gain": [[c, float(g)] for c, g in importance]}
        np.save(args.output / f"valid-{name}.npy", prediction.astype(np.float32))
        print(name, json.dumps({k: v for k, v in results[name].items() if k != "top_gain"}), flush=True)
        print(" top:", [c for c, _ in importance[:15]], flush=True)
        (args.output / "results.json").write_text(json.dumps({
            "cutoff": CUTOFF.isoformat(), "valid_rows": int(len(y_valid)), "fit_rows": int(len(y_fit)),
            "profile_columns": n_profile, "crosses": CROSSES, "univariate": univariate,
            "early_stopping_on_holdout": True, "results": results,
            "seconds": time.time() - start}, indent=2) + "\n")
        del x_fit, x_valid, train_set, valid_set, model
        gc.collect()
    np.save(args.output / "valid-row-ids.npy", valid_rows["row"].to_numpy())


if __name__ == "__main__":
    main()
