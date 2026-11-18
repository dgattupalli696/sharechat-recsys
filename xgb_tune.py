"""T034: XGBoost GPU tuning on the T029 wide request-context features.

`build` writes the fit/holdout matrices once as .npy; `worker` fits a list of named
configs (one process per GPU) and saves holdout predictions with file row ids.
Protocol matches T029: original cutoff, early stopping on the holdout.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import roc_auc_score

BASE = {
    "objective": "binary:logistic", "eval_metric": "auc", "tree_method": "hist", "device": "cuda",
    "max_depth": 0, "max_leaves": 127, "grow_policy": "lossguide", "min_child_weight": 20,
    "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.5, "lambda": 10, "seed": 42, "max_bin": 127,
}

CONFIGS = {
    "t029_gpu": {},
    "leaves255_eta02": {"max_leaves": 255, "eta": 0.02, "colsample_bytree": 0.3, "min_child_weight": 100},
    "leaves511_eta02": {"max_leaves": 511, "eta": 0.02, "colsample_bytree": 0.3, "min_child_weight": 200,
                        "alpha": 1},
    "leaves63_eta02": {"max_leaves": 63, "eta": 0.02, "colsample_bytree": 0.4, "min_child_weight": 50},
    "depth10": {"grow_policy": "depthwise", "max_depth": 10, "max_leaves": 0, "eta": 0.03,
                "colsample_bytree": 0.4, "min_child_weight": 50},
    "depth12_node": {"grow_policy": "depthwise", "max_depth": 12, "max_leaves": 0, "eta": 0.02,
                     "colsample_bytree": 0.8, "colsample_bynode": 0.4, "min_child_weight": 100, "gamma": 1},
    "bin255": {"max_bin": 255, "max_leaves": 255, "eta": 0.02, "colsample_bytree": 0.3,
               "min_child_weight": 100},
    "native_cat": {"max_leaves": 255, "eta": 0.02, "colsample_bytree": 0.3, "min_child_weight": 100,
                   "max_cat_to_onehot": 8, "max_cat_threshold": 64, "_categorical": True},
    "subsample06": {"max_leaves": 255, "eta": 0.02, "subsample": 0.6, "colsample_bytree": 0.3,
                    "min_child_weight": 100, "sampling_method": "gradient_based"},
    "rank_global": {"objective": "rank:pairwise", "max_leaves": 63, "eta": 0.02, "colsample_bytree": 0.4,
                    "min_child_weight": 50, "lambdarank_pair_method": "mean", "lambdarank_num_pair_per_sample": 8,
                    "_qid": "none"},
    "rank_request": {"objective": "rank:pairwise", "max_leaves": 63, "eta": 0.02, "colsample_bytree": 0.4,
                     "min_child_weight": 50, "lambdarank_pair_method": "mean", "lambdarank_num_pair_per_sample": 8,
                     "_qid": "request"},
    "lambda50": {"max_leaves": 255, "eta": 0.02, "colsample_bytree": 0.3, "min_child_weight": 100,
                 "lambda": 50, "alpha": 5},
}


ID_COLUMNS = ["feature5", "feature18", "feature23", "feature20"]


def train_split_columns(data, columns, rows=0):
    """Pre-cutoff / holdout values of raw train columns, in file order (matches `build_matrices`)."""
    import polars as pl

    from feature_screen import CUTOFF

    lf = pl.scan_parquet(data / "train.parquet")
    if rows:
        lf = lf.filter(pl.col("request_id").hash(7) % rows == 0)
    frame = lf.select("feature1", *columns).collect()
    is_valid = (frame["feature1"] >= CUTOFF).to_numpy()
    return frame.filter(~pl.Series(is_valid)).drop("feature1"), frame.filter(pl.Series(is_valid)).drop("feature1")


def cosine_frame(path, rows=0, fit_only=False):
    """User x item cosines (T035 `cos` set) plus within-request rank/deviation, in file order."""
    import polars as pl

    from embed_screen import COS_PAIRS, unit_vectors

    lf = pl.scan_parquet(path)
    if rows:
        lf = lf.filter(pl.col("request_id").hash(7) % rows == 0)
    if fit_only:
        from feature_screen import CUTOFF

        lf = lf.filter(pl.col("feature1") < CUTOFF)
    vectors = lf.select("request_id", *dict.fromkeys(c for pair in COS_PAIRS for c in pair)).collect()
    extra = {}
    for u, i in COS_PAIRS:
        extra[f"cos_{u[7:]}_{i[7:]}"] = (unit_vectors(vectors[u]) * unit_vectors(vectors[i])).sum(1)
    names = list(extra)
    frame = pl.DataFrame({"request_id": vectors["request_id"], **extra})
    frame = frame.with_columns([
        e for c in names for e in (
            (pl.col(c).rank("average").over("request_id") / pl.len().over("request_id")).alias(f"rq_rank_{c}"),
            (pl.col(c) - pl.col(c).mean().over("request_id")).alias(f"rq_dev_{c}"))]).drop("request_id")
    return frame.to_numpy().astype(np.float32), frame.columns


def entity_frame(data, rows=0):
    """Label-free item/creator features over train+test (file order): upload age, past exposure
    counts, and the entity's mean centered within-request rank of top predictors. Backward-only
    windows so rows near the end of test are not truncated."""
    import polars as pl

    def scan(name):
        lf = pl.scan_parquet(data / f"{name}.parquet")
        if rows:
            lf = lf.filter(pl.col("request_id").hash(7) % rows == 0)
        return lf.select("request_id", "feature1", "feature18", "feature23", "feature58", "feature59",
                         "feature54", "feature409", "feature506").collect()

    train, test = scan("train"), scan("test")
    frame = pl.concat([train, test])
    hours = frame["feature1"].dt.epoch("ms").to_numpy() / 3.6e6
    columns = {"age_up": hours - frame["feature58"].to_numpy() / 3600.0,
               "age_up59": hours - frame["feature59"].to_numpy() / 3.6e6}
    hours = hours - hours.min()
    size = pl.len().over("request_id")
    ranks = frame.select([((pl.col(c).rank("average").over("request_id") - (size + 1) / 2) / size).alias(c)
                          for c in ["feature54", "feature409", "feature506"]])
    for key in ["feature18", "feature23"]:
        codes = frame[key].cast(pl.Categorical).to_physical().to_numpy().astype(np.int64)
        suffix = key[-2:]
        stamp = codes * 1e4 + hours
        ordered = np.sort(stamp)
        columns[f"since_first_{suffix}"] = hours - pl.DataFrame({"k": codes, "t": hours}).select(
            pl.col("t").min().over("k"))["t"].to_numpy()
        for window in (1, 6, 24):
            columns[f"nb{window}_{suffix}"] = (np.searchsorted(ordered, stamp, "left")
                                               - np.searchsorted(ordered, stamp - window, "left"))
        means = ranks.with_columns(k=codes).select(pl.all().exclude("k").mean().over("k"))
        for c in means.columns:
            columns[f"ent_crk_{c}_{suffix}"] = means[c].to_numpy()
    values = np.column_stack(list(columns.values())).astype(np.float32)
    return values[: train.height], values[train.height:], list(columns)


def age_frame(path, rows=0):
    """Log item age (hours since upload, from feature58 s and feature59 ms) with within-request
    rank/deviation; replaces the absolute upload timestamps, which drift between train and test."""
    import polars as pl

    lf = pl.scan_parquet(path)
    if rows:
        lf = lf.filter(pl.col("request_id").hash(7) % rows == 0)
    ts = pl.col("feature1").dt.epoch("s").cast(pl.Float64)
    frame = lf.select(
        "feature1",
        ((ts - pl.col("feature58").cast(pl.Float64)).clip(1.0, None) / 3600).log1p().alias("log_age"),
        ((ts - pl.col("feature59").cast(pl.Float64) / 1000).clip(1.0, None) / 3600).log1p().alias("log_age59"),
        "request_id").with_columns(
        (pl.col("log_age").rank().over("request_id") / pl.len().over("request_id")).alias("rq_rank_age"),
        (pl.col("log_age") - pl.col("log_age").mean().over("request_id")).alias("rq_dev_age")).collect()
    names = ["log_age", "log_age59", "rq_rank_age", "rq_dev_age"]
    return frame.select(names).to_numpy().astype(np.float32), frame["feature1"], names


def replace_upload_time(args, x_fit, x_valid, columns):
    from feature_screen import CUTOFF

    values, stamp, names = age_frame(args.data / "train.parquet", args.rows)
    is_valid = (stamp >= CUTOFF).to_numpy()
    keep = [i for i, c in enumerate(columns) if c not in ("feature58", "feature59")]
    assert len(keep) == len(columns) - 2, "expected raw feature58/feature59 columns"
    return (np.hstack([x_fit[:, keep], values[~is_valid]]), np.hstack([x_valid[:, keep], values[is_valid]]),
            [columns[i] for i in keep] + names)


def cosine_columns(data, rows):
    import polars as pl

    from feature_screen import CUTOFF

    values, names = cosine_frame(data / "train.parquet", rows)
    lf = pl.scan_parquet(data / "train.parquet")
    if rows:
        lf = lf.filter(pl.col("request_id").hash(7) % rows == 0)
    is_valid = (lf.select("feature1").collect()["feature1"] >= CUTOFF).to_numpy()
    return values[~is_valid], values[is_valid], names


def build(args):
    from classic_screen import build_matrices

    start = time.time()
    wide = json.loads(args.wide_from.read_text())["wide_features"]
    x_fit, y_fit, x_valid, y_valid, row_ids, columns, categorical = build_matrices(args.data, wide, args.rows)
    if args.cosines:
        c_fit, c_valid, names = cosine_columns(args.data, args.rows)
        x_fit, x_valid, columns = np.hstack([x_fit, c_fit]), np.hstack([x_valid, c_valid]), columns + names
        del c_fit, c_valid
    if args.entity:
        from feature_screen import CUTOFF
        import polars as pl

        values, _, names = entity_frame(args.data, args.rows)
        lf = pl.scan_parquet(args.data / "train.parquet")
        if args.rows:
            lf = lf.filter(pl.col("request_id").hash(7) % args.rows == 0)
        is_valid = (lf.select("feature1").collect()["feature1"] >= CUTOFF).to_numpy()
        x_fit, x_valid = np.hstack([x_fit, values[~is_valid]]), np.hstack([x_valid, values[is_valid]])
        columns = columns + names
        del values
    if args.age:
        x_fit, x_valid, columns = replace_upload_time(args, x_fit, x_valid, columns)
    args.matrices.mkdir(parents=True, exist_ok=True)
    for name, value in dict(x_fit=x_fit, y_fit=y_fit, x_valid=x_valid, y_valid=y_valid, row_ids=row_ids).items():
        np.save(args.matrices / f"{name}.npy", value)
    (args.matrices / "columns.json").write_text(json.dumps({"columns": columns, "categorical": categorical}))
    print(f"matrices {x_fit.shape} / {x_valid.shape} in {time.time() - start:.0f}s", flush=True)


def worker(args):
    import xgboost as xgb

    m = args.matrices
    x_fit, x_valid = np.load(m / "x_fit.npy", mmap_mode="r"), np.load(m / "x_valid.npy", mmap_mode="r")
    y_fit, y_valid, row_ids = np.load(m / "y_fit.npy"), np.load(m / "y_valid.npy"), np.load(m / "row_ids.npy")
    meta = json.loads((m / "columns.json").read_text())
    feature_types = ["c" if i in set(meta["categorical"]) else "q" for i in range(len(meta["columns"]))]
    args.output.mkdir(parents=True, exist_ok=True)
    for name in args.configs.split(","):
        began = time.time()
        params = {**BASE, **CONFIGS[name], "device": args.device}
        categorical = params.pop("_categorical", False)
        qid = params.pop("_qid", None)
        types = dict(feature_types=feature_types, enable_categorical=True) if categorical else {}
        if qid == "request":
            fit_rq, _ = train_split_columns(args.data, ["request_id"], args.rows)
            codes = fit_rq["request_id"].to_numpy()
            types["qid"] = np.cumsum(np.r_[True, codes[1:] != codes[:-1]]).astype(np.uint32)
        elif qid == "none":
            types["qid"] = np.zeros(len(y_fit), dtype=np.uint32)
        if args.pseudo:
            # Self-training: append the unlabeled rows with teacher soft labels; early stopping still uses true labels.
            z = np.load(args.pseudo)
            pos = {r: i for i, r in enumerate(z["row_ids"])}
            soft = z["probabilities"][[pos[r] for r in row_ids]].astype(np.float32)
            fit = xgb.QuantileDMatrix(np.vstack([x_fit, x_valid]), np.concatenate([y_fit, soft]),
                                      weight=np.r_[np.ones(len(y_fit)), np.full(len(soft), args.pseudo_weight)],
                                      max_bin=params["max_bin"], **types)
        else:
            fit = xgb.QuantileDMatrix(np.asarray(x_fit), y_fit, max_bin=params["max_bin"], **types)
        types.pop("qid", None)
        valid = xgb.QuantileDMatrix(np.asarray(x_valid), y_valid, ref=fit, max_bin=params["max_bin"], **types)
        model = xgb.train(params, fit, args.rounds, evals=[(valid, "holdout")],
                          early_stopping_rounds=300, verbose_eval=500)
        predictions = model.predict(valid, iteration_range=(0, model.best_iteration + 1))
        if qid:
            predictions = 1 / (1 + np.exp(-predictions))
        auc = float(roc_auc_score(y_valid, predictions))
        np.savez_compressed(args.output / f"{name}-holdout.npz", row_ids=row_ids, y_true=y_valid,
                            probabilities=predictions.astype(np.float32))
        gain = model.get_score(importance_type="total_gain")
        top = sorted(gain.items(), key=lambda kv: -kv[1])[:30]
        result = {"holdout_auc": auc, "best_iteration": int(model.best_iteration),
                  "seconds": time.time() - began, "params": params,
                  "top_gain": [[meta["columns"][int(k[1:])], v] for k, v in top]}
        (args.output / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(name, json.dumps({k: result[k] for k in ["holdout_auc", "best_iteration", "seconds"]}), flush=True)
        del fit, valid, model


def catboost_worker(args):
    """CatBoost (GPU) on the same matrices; low-cardinality codes and optional raw IDs as categoricals."""
    from catboost import CatBoostClassifier, FeaturesData, Pool

    m = args.matrices
    meta = json.loads((m / "columns.json").read_text())
    y_fit, y_valid, row_ids = np.load(m / "y_fit.npy"), np.load(m / "y_valid.npy"), np.load(m / "row_ids.npy")
    categorical = meta["categorical"]
    dense = [i for i in range(len(meta["columns"])) if i not in set(categorical)]
    ids = ID_COLUMNS if args.catboost_ids else []
    raw = train_split_columns(args.data, ids, args.rows) if ids else None

    def pool(split, y, part):
        x = np.load(m / f"x_{split}.npy", mmap_mode="r")
        codes = np.nan_to_num(np.asarray(x[:, categorical]), nan=-1).astype(np.int64).astype(str)
        if ids:
            codes = np.hstack([codes, part.fill_null("__NA__").to_numpy().astype(str)])
        return Pool(FeaturesData(num_feature_data=np.ascontiguousarray(x[:, dense], dtype=np.float32),
                                 cat_feature_data=codes.astype(object)), y)

    began = time.time()
    fit = pool("fit", y_fit, raw[0] if ids else None)
    valid = pool("valid", y_valid, raw[1] if ids else None)
    name = "catboost_ids" if ids else "catboost"
    model = CatBoostClassifier(
        iterations=args.rounds, learning_rate=0.06, depth=8, border_count=128, l2_leaf_reg=10,
        eval_metric="AUC", metric_period=50, od_type="Iter", od_wait=300, random_seed=42, verbose=500,
        use_best_model=True, task_type="GPU" if args.device.startswith("cuda") else "CPU",
        **({"devices": "0"} if args.device.startswith("cuda") else {"thread_count": 6}),
        max_ctr_complexity=2, gpu_ram_part=0.9 if args.device.startswith("cuda") else None,
    )
    model.fit(fit, eval_set=valid)
    predictions = model.predict_proba(valid)[:, 1]
    auc = float(roc_auc_score(y_valid, predictions))
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / f"{name}-holdout.npz", row_ids=row_ids, y_true=y_valid,
                        probabilities=predictions.astype(np.float32))
    model.save_model(str(args.output / f"{name}.cbm"))
    result = {"holdout_auc": auc, "best_iteration": int(model.get_best_iteration()),
              "seconds": time.time() - began, "ids": ids}
    (args.output / f"{name}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(name, json.dumps(result), flush=True)


def catboost_refit(args):
    """Fixed-iteration CatBoost (+raw IDs) on every labeled row; `--configs seed:iterations,...`."""
    import polars as pl
    from catboost import CatBoostClassifier, FeaturesData, Pool

    m = args.matrices
    meta = json.loads((m / "columns.json").read_text())
    y_train, test_ids = np.load(m / "y_train.npy"), np.load(m / "test_ids.npy")
    categorical = meta["categorical"]
    dense = [i for i in range(len(meta["columns"])) if i not in set(categorical)]

    def raw(name, extra):
        lf = pl.scan_parquet(args.data / f"{name}.parquet")
        if args.rows:
            lf = lf.filter(pl.col("request_id").hash(7) % args.rows == 0)
        return lf.select(extra, *ID_COLUMNS).collect()

    def pool(split, ids, y=None):
        x = np.load(m / f"x_{split}.npy", mmap_mode="r")
        codes = np.nan_to_num(np.asarray(x[:, categorical]), nan=-1).astype(np.int64).astype(str)
        codes = np.hstack([codes, ids.select(ID_COLUMNS).fill_null("__NA__").to_numpy().astype(str)])
        return Pool(FeaturesData(num_feature_data=np.ascontiguousarray(x[:, dense], dtype=np.float32),
                                 cat_feature_data=codes.astype(object)), y)

    train_ids, test_raw = raw("train", "watched"), raw("test", "id")
    if not np.array_equal(train_ids["watched"].to_numpy(), y_train):
        raise ValueError("Train ID columns are not aligned with the matrices")
    if not np.array_equal(test_raw["id"].to_numpy(), test_ids):
        raise ValueError("Test ID columns are not aligned with the matrices")
    template = None
    if (args.data / "sample_submission.csv").is_file() and not args.rows:
        template = pl.read_csv(args.data / "sample_submission.csv")
        if not np.array_equal(template["id"].to_numpy(), test_ids):
            raise ValueError("Test ids differ from the submission template")
    train, test = pool("train", train_ids, y_train), pool("test", test_raw)
    args.output.mkdir(parents=True, exist_ok=True)
    gpu = args.device.startswith("cuda")
    for item in args.configs.split(","):
        began = time.time()
        seed, iterations = (int(v) for v in item.split(":"))
        tag = f"catboost_ids-s{seed}"
        model = CatBoostClassifier(
            iterations=iterations, learning_rate=0.06, depth=8, border_count=128, l2_leaf_reg=10,
            random_seed=seed, verbose=500, task_type="GPU" if gpu else "CPU",
            **({"devices": "0", "gpu_ram_part": 0.9} if gpu else {"thread_count": 6}), max_ctr_complexity=2)
        model.fit(train)
        predictions = model.predict_proba(test)[:, 1].astype(np.float32)
        if not np.isfinite(predictions).all():
            raise ValueError(f"Nonfinite predictions: {tag}")
        np.savez_compressed(args.output / f"{tag}-test.npz", ids=test_ids, probabilities=predictions)
        model.save_model(str(args.output / f"{tag}.cbm"))
        if template is not None:
            template.with_columns(pl.Series("watched", predictions)).write_csv(args.output / f"submission-{tag}.csv")
        result = {"iterations": iterations, "seed": seed, "training_rows": int(len(y_train)),
                  "ids": ID_COLUMNS, "seconds": time.time() - began}
        (args.output / f"{tag}-refit.json").write_text(json.dumps(result, indent=2) + "\n")
        print(tag, json.dumps(result), flush=True)
        del model


def kfold(args):
    """5-fold (grouped by request) fits on every labeled row with in-fold early stopping.

    `--configs` takes XGBoost config names and/or `catboost_ids`. Saves out-of-fold
    predictions on the train rows plus per-fold and fold-mean test predictions.
    """
    import polars as pl
    from sklearn.model_selection import GroupKFold

    m = args.matrices
    meta = json.loads((m / "columns.json").read_text())
    y_train, test_ids = np.load(m / "y_train.npy"), np.load(m / "test_ids.npy")
    lf = pl.scan_parquet(args.data / "train.parquet")
    test_lf = pl.scan_parquet(args.data / "test.parquet")
    if args.rows:
        lf = lf.filter(pl.col("request_id").hash(7) % args.rows == 0)
        test_lf = test_lf.filter(pl.col("request_id").hash(7) % args.rows == 0)
    train_raw = lf.select("watched", "request_id", *ID_COLUMNS).collect()
    if not np.array_equal(train_raw["watched"].to_numpy(), y_train):
        raise ValueError("Train rows are not aligned with the matrices")
    folds = list(GroupKFold(args.folds).split(y_train, groups=train_raw["request_id"].to_numpy()))
    x_train, x_test = np.load(m / "x_train.npy", mmap_mode="r"), np.load(m / "x_test.npy", mmap_mode="r")
    template = None
    if (args.data / "sample_submission.csv").is_file() and not args.rows:
        template = pl.read_csv(args.data / "sample_submission.csv")
        if not np.array_equal(template["id"].to_numpy(), test_ids):
            raise ValueError("Test ids differ from the submission template")
    args.output.mkdir(parents=True, exist_ok=True)
    for name in args.configs.split(","):
        began = time.time()
        oof = np.full(len(y_train), np.nan, dtype=np.float32)
        tests, info = [], []
        if name == "catboost_ids":
            from catboost import CatBoostClassifier, FeaturesData, Pool

            test_raw = test_lf.select("id", *ID_COLUMNS).collect()
            if not np.array_equal(test_raw["id"].to_numpy(), test_ids):
                raise ValueError("Test rows are not aligned with the matrices")
            categorical = meta["categorical"]
            dense = [i for i in range(len(meta["columns"])) if i not in set(categorical)]

            def codes(x, raw):
                low = np.nan_to_num(np.asarray(x[:, categorical]), nan=-1).astype(np.int64).astype(str)
                return np.hstack([low, raw.select(ID_COLUMNS).fill_null("__NA__").to_numpy().astype(str)]).astype(object)

            train_codes, test_codes = codes(x_train, train_raw), codes(x_test, test_raw)
            test_pool = Pool(FeaturesData(num_feature_data=np.ascontiguousarray(x_test[:, dense], dtype=np.float32),
                                          cat_feature_data=test_codes))
            gpu = args.device.startswith("cuda")
            for k, (a, v) in enumerate(folds):
                def pool(rows):
                    return Pool(FeaturesData(num_feature_data=np.ascontiguousarray(x_train[rows][:, dense], dtype=np.float32),
                                             cat_feature_data=train_codes[rows]), y_train[rows])
                model = CatBoostClassifier(
                    iterations=args.rounds, learning_rate=0.06, depth=8, border_count=128, l2_leaf_reg=10,
                    eval_metric="AUC", metric_period=50, od_type="Iter", od_wait=300, random_seed=42 + k,
                    verbose=500, use_best_model=True, task_type="GPU" if gpu else "CPU",
                    **({"devices": "0", "gpu_ram_part": 0.9} if gpu else {"thread_count": 6}), max_ctr_complexity=2)
                fit_pool, valid_pool = pool(a), pool(v)
                model.fit(fit_pool, eval_set=valid_pool)
                oof[v] = model.predict_proba(valid_pool)[:, 1]
                tests.append(model.predict_proba(test_pool)[:, 1].astype(np.float32))
                info.append({"fold": k, "best_iteration": int(model.get_best_iteration()),
                             "fold_auc": float(roc_auc_score(y_train[v], oof[v])), "seconds": time.time() - began})
                print(name, json.dumps(info[-1]), flush=True)
                del model, fit_pool, valid_pool
        else:
            import xgboost as xgb

            params = {**BASE, **CONFIGS[name], "device": args.device}
            if params.pop("_categorical", False) or params.pop("_qid", None):
                raise ValueError("Only plain XGBoost configs are supported in kfold")
            for k, (a, v) in enumerate(folds):
                params["seed"] = 42 + k
                fit = xgb.QuantileDMatrix(x_train[a], y_train[a], max_bin=params["max_bin"])
                valid = xgb.QuantileDMatrix(x_train[v], y_train[v], max_bin=params["max_bin"], ref=fit)
                model = xgb.train(params, fit, args.rounds, evals=[(valid, "fold")],
                                  early_stopping_rounds=300, verbose_eval=500)
                best = model.best_iteration
                oof[v] = model.predict(valid, iteration_range=(0, best + 1))
                test = xgb.QuantileDMatrix(x_test, max_bin=params["max_bin"], ref=fit)
                tests.append(model.predict(test, iteration_range=(0, best + 1)).astype(np.float32))
                info.append({"fold": k, "best_iteration": int(best),
                             "fold_auc": float(roc_auc_score(y_train[v], oof[v])), "seconds": time.time() - began})
                print(name, json.dumps(info[-1]), flush=True)
                del fit, valid, test, model
        mean = np.mean(tests, 0).astype(np.float32)
        if not np.isfinite(mean).all() or np.isnan(oof).any():
            raise ValueError(f"Incomplete or nonfinite predictions: {name}")
        np.savez_compressed(args.output / f"{name}-kfold.npz", ids=test_ids, probabilities=mean,
                            fold_probabilities=np.stack(tests), oof=oof, y_train=y_train)
        if template is not None:
            template.with_columns(pl.Series("watched", mean)).write_csv(args.output / f"submission-{name}-kfold.csv")
        result = {"folds": info, "oof_auc": float(roc_auc_score(y_train, oof)), "seconds": time.time() - began}
        (args.output / f"{name}-kfold.json").write_text(json.dumps(result, indent=2) + "\n")
        print(name, "oof", result["oof_auc"], flush=True)


def build_full(args):
    from classic_screen import full_matrices

    start = time.time()
    wide = json.loads(args.wide_from.read_text())["wide_features"]
    x_train, y_train, x_test, test_ids, columns, categorical = full_matrices(args.data, wide, args.rows, args.fit_only)
    if args.cosines:
        c_train, names = cosine_frame(args.data / "train.parquet", args.rows, args.fit_only)
        x_train, columns = np.hstack([x_train, c_train]), columns + names
        del c_train
        c_test, _ = cosine_frame(args.data / "test.parquet", args.rows)
        x_test = np.hstack([x_test, c_test])
        del c_test
    args.matrices.mkdir(parents=True, exist_ok=True)
    for name, value in dict(x_train=x_train, y_train=y_train, x_test=x_test, test_ids=test_ids).items():
        np.save(args.matrices / f"{name}.npy", value)
    (args.matrices / "columns.json").write_text(json.dumps({"columns": columns, "categorical": categorical}))
    print(f"full matrices {x_train.shape} / {x_test.shape} in {time.time() - start:.0f}s", flush=True)


def refit(args):
    """Fixed-round fits on every labeled row; `--configs name[@seed]:rounds,...`; test predictions saved."""
    import polars as pl
    import xgboost as xgb

    m = args.matrices
    y_train, test_ids = np.load(m / "y_train.npy"), np.load(m / "test_ids.npy")
    meta = json.loads((m / "columns.json").read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    template = None
    if (args.data / "sample_submission.csv").is_file() and not args.rows:
        template = pl.read_csv(args.data / "sample_submission.csv")
        if not np.array_equal(template["id"].to_numpy(), test_ids):
            raise ValueError("Test ids differ from the submission template")
    for item in args.configs.split(","):
        began = time.time()
        spec, rounds = item.split(":")
        name, _, seed = spec.partition("@")
        tag = f"{name}-s{seed}" if seed else name
        params = {**BASE, **CONFIGS[name], "device": args.device}
        if seed:
            params["seed"] = int(seed)
        if params.pop("_categorical", False):
            raise ValueError("Native categorical configs are not supported in refit")
        x_train = np.load(m / "x_train.npy", mmap_mode="r")
        if args.pseudo:
            z = np.load(args.pseudo)
            order = np.argsort(z["ids"])
            pos = order[np.searchsorted(z["ids"], test_ids, sorter=order)]
            if not np.array_equal(z["ids"][pos], test_ids):
                raise ValueError("Pseudo-label ids differ from test ids")
            soft = z["probabilities"][pos].astype(np.float32)
            weight = np.concatenate([np.ones(len(y_train), np.float32),
                                     np.full(len(soft), args.pseudo_weight, np.float32)])
            x_all = np.vstack([x_train, np.load(m / "x_test.npy", mmap_mode="r")])
            train = xgb.QuantileDMatrix(x_all, np.concatenate([y_train.astype(np.float32), soft]),
                                        weight=weight, max_bin=params["max_bin"])
            del x_all
        else:
            train = xgb.QuantileDMatrix(x_train, y_train, max_bin=params["max_bin"])
        model = xgb.train(params, train, int(rounds))
        test = xgb.QuantileDMatrix(np.load(m / "x_test.npy", mmap_mode="r"), max_bin=params["max_bin"], ref=train)
        predictions = model.predict(test).astype(np.float32)
        del train, test
        if not np.isfinite(predictions).all():
            raise ValueError(f"Nonfinite predictions: {tag}")
        np.savez_compressed(args.output / f"{tag}-test.npz", ids=test_ids, probabilities=predictions)
        model.save_model(args.output / f"{tag}.ubj")
        if template is not None:
            template.with_columns(pl.Series("watched", predictions)).write_csv(args.output / f"submission-{tag}.csv")
        result = {"rounds": int(rounds), "training_rows": int(len(y_train)), "pseudo": str(args.pseudo or ""),
                  "pseudo_weight": args.pseudo_weight if args.pseudo else 0, "features": len(meta["columns"]),
                  "seconds": time.time() - began, "params": params}
        (args.output / f"{tag}-refit.json").write_text(json.dumps(result, indent=2) + "\n")
        print(tag, json.dumps({k: result[k] for k in ["rounds", "training_rows", "seconds"]}), flush=True)
        del model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["build", "worker", "build_full", "refit", "catboost", "catboost_refit", "kfold"])
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--wide-from", type=Path, default=Path("artifacts/local-t026/results.json"))
    parser.add_argument("--matrices", type=Path, default=Path("artifacts/local-t034/matrices"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/local-t034"))
    parser.add_argument("--configs", default=",".join(CONFIGS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cosines", action="store_true", help="Append T035 user x item cosine features")
    parser.add_argument("--entity", action="store_true", help="build: append label-free item/creator features")
    parser.add_argument("--age", action="store_true", help="build: replace feature58/59 with log item age")
    parser.add_argument("--pseudo", type=Path, help="worker: npz (row_ids, probabilities) soft labels for holdout rows; refit: npz (ids, probabilities) for test rows")
    parser.add_argument("--pseudo-weight", type=float, default=1.0)
    parser.add_argument("--rounds", type=int, default=20000)
    parser.add_argument("--catboost-ids", action="store_true", help="catboost: add raw user/item/ID columns")
    parser.add_argument("--fit-only", action="store_true", help="build_full: train rows before the cutoff only")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--rows", type=int, default=0, help="Keep 1/N of requests for smoke tests")
    args = parser.parse_args()
    {"build": build, "worker": worker, "build_full": build_full, "refit": refit,
     "catboost": catboost_worker, "catboost_refit": catboost_refit, "kfold": kfold}[args.mode](args)


if __name__ == "__main__":
    main()
