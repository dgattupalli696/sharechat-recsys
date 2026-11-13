"""T035 local LightGBM screen: item-embedding request context and similar-video watch rates.

Row-level vectors feature40-46 are near-constant per item (cos 0.999 to the item mean),
so each item gets a label-free mean embedding from train+test. Variants on top of the
T023/T031 request-context control:
- cos: user(26-32) x item(40-46) cosines plus their within-request rank/deviation;
- embctx: cosine to the leave-one-out request mean item embedding, request embedding
  diversity, distinct creators/categories in the request;
- knn: out-of-fold watch rate of the top-k most similar other items (embedding kNN).
Scores the original 307,528-row holdout; request_id is only used for grouping.
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
from sklearn.model_selection import GroupKFold

from cf_screen import within_request_auc
from feature_screen import CUTOFF, ITEM, LOW_CATEGORICAL, RANKED, request_features, scalar_numeric

USER_VECTORS = [f"feature{i}" for i in range(26, 33)]
ITEM_VECTORS = [f"feature{i}" for i in range(40, 47)]
COS_PAIRS = [(u, i) for u, i in zip(USER_VECTORS, ITEM_VECTORS)] + [
    ("feature32", "feature43"), ("feature30", "feature46"), ("feature31", "feature44")]
KNN_VECTORS = ["feature40", "feature43", "feature46"]
CREATOR, CATEGORY = "feature23", "feature19"
SMOOTHING = 20.0


def unit_vectors(series):
    values = series.fill_null([0.0] * 32).list.to_array(32).to_numpy().astype(np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-6)


def item_neighbours(data, rows, k, chunk=512):
    """Label-free item mean embeddings over train+test and their top-k cosine neighbours."""
    parts = []
    for name in ["train.parquet", "test.parquet"]:
        lf = pl.scan_parquet(data / name)
        if rows:
            lf = lf.filter(pl.col("request_id").hash(7) % rows == 0)
        parts.append(lf.select(ITEM, *KNN_VECTORS).collect())
    frame = pl.concat(parts)
    del parts
    blocks = [unit_vectors(frame[c]) for c in KNN_VECTORS]
    keys, inverse = np.unique(frame[ITEM].to_numpy(), return_inverse=True)
    counts = np.bincount(inverse).astype(np.float32)
    means = []
    for block in blocks:
        total = np.zeros((len(keys), 32), dtype=np.float64)
        np.add.at(total, inverse, block)
        mean = total / counts[:, None]
        means.append(mean / np.maximum(np.linalg.norm(mean, axis=1, keepdims=True), 1e-6))
    embedding = (np.hstack(means) / np.sqrt(len(means))).astype(np.float32)
    del frame, blocks
    k = min(k, len(keys) - 1)
    index = np.empty((len(keys), k), dtype=np.int32)
    weight = np.empty((len(keys), k), dtype=np.float32)
    for start in range(0, len(keys), chunk):
        sims = embedding[start:start + chunk] @ embedding.T
        sims[np.arange(len(sims)), np.arange(start, start + len(sims))] = -np.inf
        top = np.argpartition(-sims, k, axis=1)[:, :k]
        index[start:start + chunk] = top
        weight[start:start + chunk] = np.take_along_axis(sims, top, axis=1)
    return pl.DataFrame({ITEM: keys}).with_row_index("item_idx"), index, weight, embedding


def knn_rates(item_idx, index, weight, s, n, prior, temperature=0.05):
    """Similarity-weighted neighbour watch rate (softmax over cosine) and its support."""
    w = np.exp((weight[item_idx] - weight[item_idx].max(axis=1, keepdims=True)) / temperature)
    ns, nn = n[index[item_idx]], s[index[item_idx]]
    rate = ((w * nn).sum(1) + SMOOTHING * prior) / ((w * ns).sum(1) + SMOOTHING)
    support = (w * ns).sum(1) / w.sum(1)
    seen = ns > 0
    best = np.where(seen, weight[item_idx], -np.inf).max(1)
    return rate, np.log1p(support), np.where(np.isfinite(best), best, np.nan)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/local-t035"))
    parser.add_argument("--variants", default="control,cos,embctx,knn")
    parser.add_argument("--neighbours", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--rounds", type=int, default=4000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--rows", type=int, default=0, help="Keep 1/N of requests for smoke tests")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.time()
    items, index, weight, embedding = item_neighbours(args.data, args.rows, args.neighbours)
    print(f"item kNN ready ({len(items)} items) {time.time() - start:.0f}s", flush=True)

    schema = pl.scan_parquet(args.data / "train.parquet").collect_schema()
    numeric = scalar_numeric(schema)
    lf = pl.scan_parquet(args.data / "train.parquet").with_row_index("row")
    if args.rows:
        lf = lf.filter(pl.col("request_id").hash(7) % args.rows == 0)
    keys = sorted(set([ITEM, CREATOR, CATEGORY] + LOW_CATEGORICAL), key=lambda c: int(c[7:]))
    train = lf.select("row", "watched", "request_id", "feature1", "feature3", *keys,
                      *[c for c in numeric if c not in keys]).collect()
    rq, rq_names = request_features(train.select("row", "request_id", ITEM, *RANKED))
    train = train.join(rq, on="row", how="left")
    del rq
    train = train.join(items.with_columns(pl.col(ITEM).cast(train.schema[ITEM])), on=ITEM, how="left")
    extra = {}

    # cos: user x item cosine similarities and their within-request rank/deviation.
    vectors = lf.select(*dict.fromkeys(c for pair in COS_PAIRS for c in pair)).collect()
    units = {c: unit_vectors(vectors[c]) for c in vectors.columns}
    del vectors
    cos_names = []
    for u, i in COS_PAIRS:
        name = f"cos_{u[7:]}_{i[7:]}"
        extra[name] = (units[u] * units[i]).sum(1)
        cos_names.append(name)
    frame = pl.DataFrame({"request_id": train["request_id"], **extra})
    ranked = frame.select([
        e for c in cos_names for e in (
            (pl.col(c).rank("average").over("request_id") / pl.len().over("request_id")).alias(f"rq_rank_{c}"),
            (pl.col(c) - pl.col(c).mean().over("request_id")).alias(f"rq_dev_{c}"))])
    for c in ranked.columns:
        extra[c] = ranked[c].to_numpy()
    cos_names += ranked.columns
    del ranked, frame

    # embctx: similarity to the rest of the request in item-embedding space.
    requests = train["request_id"].to_numpy()
    _, request_code = np.unique(requests, return_inverse=True)
    size = np.bincount(request_code).astype(np.float32)[request_code]
    ctx_names = []
    for c in ["feature40", "feature43", "feature46"]:
        u = units.get(c)
        if u is None:
            u = unit_vectors(lf.select(c).collect()[c])
        total = np.zeros((request_code.max() + 1, 32), dtype=np.float64)
        np.add.at(total, request_code, u)
        others = total[request_code] - u
        norm = np.linalg.norm(others, axis=1)
        loo = np.where(size > 1, (u * others).sum(1) / np.maximum(norm, 1e-6), np.nan)
        sq = (total ** 2).sum(1)[request_code]
        diversity = np.where(size > 1, (sq - size) / np.maximum(size * (size - 1), 1), np.nan)
        extra[f"emb_loo_{c}"], extra[f"emb_div_{c}"] = loo.astype(np.float32), diversity.astype(np.float32)
        extra[f"emb_lift_{c}"] = (loo - diversity).astype(np.float32)
        ctx_names += [f"emb_loo_{c}", f"emb_div_{c}", f"emb_lift_{c}"]
    del units
    counts = train.select(
        pl.col(CREATOR).n_unique().over("request_id").alias("rq_creators"),
        pl.col(CATEGORY).n_unique().over("request_id").alias("rq_categories"),
        pl.len().over(["request_id", CREATOR]).alias("rq_creator_dup"),
        pl.len().over(["request_id", CATEGORY]).alias("rq_category_dup"))
    for c in counts.columns:
        extra[c] = counts[c].to_numpy().astype(np.float32)
    extra["rq_creator_share"] = extra["rq_creator_dup"] / size
    ctx_names += [*counts.columns, "rq_creator_share"]
    del counts

    # knn: out-of-fold neighbour watch rates (neighbour lists exclude the item itself).
    is_valid = (train["feature1"] >= CUTOFF).to_numpy()
    y = train["watched"].to_numpy().astype(np.float64)
    item_idx = train["item_idx"].to_numpy().astype(np.int64)
    fit_idx, valid_idx = np.flatnonzero(~is_valid), np.flatnonzero(is_valid)
    prior = float(y[fit_idx].mean())
    knn_names = ["knn_rate", "knn_support", "knn_best_sim", "item_rate", "item_support", "knn_minus_item"]
    for name in knn_names:
        extra[name] = np.full(len(train), np.nan, dtype=np.float32)

    def fill(source, target):
        s = np.bincount(item_idx[source], y[source], minlength=len(items))
        n = np.bincount(item_idx[source], minlength=len(items)).astype(np.float64)
        rate, support, best = knn_rates(item_idx[target], index, weight, s, n, prior)
        own = (s[item_idx[target]] + SMOOTHING * prior) / (n[item_idx[target]] + SMOOTHING)
        for name, value in zip(knn_names, [rate, support, best, own, np.log1p(n[item_idx[target]]), rate - own]):
            extra[name][target] = value
    groups = train["request_id"].to_numpy()[fit_idx]
    for a, b in GroupKFold(5).split(fit_idx, groups=groups):
        fill(fit_idx[a], fit_idx[b])
    fill(fit_idx, valid_idx)
    print(f"features ready {time.time() - start:.0f}s", flush=True)

    categories = {c: train.filter(pl.Series(~is_valid))[c].drop_nulls().unique().sort() for c in LOW_CATEGORICAL}
    base = ["feature3", *[c for c in numeric if c not in categories], *LOW_CATEGORICAL, *rq_names]
    groups_ = {"control": base, "cos": base + cos_names, "embctx": base + cos_names + ctx_names,
               "knn": base + cos_names + ctx_names + knn_names}
    y_fit, y_valid = y[fit_idx], y[valid_idx]
    univariate = {}
    for name in cos_names[:10] + ctx_names + knn_names:
        v = extra[name][valid_idx]
        ok = np.isfinite(v)
        auc = roc_auc_score(y_valid[ok], v[ok])
        univariate[name] = round(float(max(auc, 1 - auc)), 4)
    print("univariate holdout AUC (max of +/-):", json.dumps(univariate), flush=True)
    item_seen = np.bincount(item_idx[fit_idx], minlength=len(items))[item_idx[valid_idx]] > 0

    def matrix(rows, columns):
        out = np.empty((len(rows), len(columns)), dtype=np.float32)
        for j, c in enumerate(columns):
            if c in extra:
                out[:, j] = extra[c][rows]
            elif c in categories:
                out[:, j] = train.select(pl.col(c).cast(pl.Enum(categories[c].to_list()), strict=False)
                                         .to_physical().cast(pl.Float32))[c].to_numpy()[rows]
            else:
                out[:, j] = train[c].cast(pl.Float32).to_numpy()[rows]
        return out

    results = {}
    valid_requests = train["request_id"][valid_idx]
    for name in args.variants.split(","):
        columns = groups_[name]
        began = time.time()
        x_fit, x_valid = matrix(fit_idx, columns), matrix(valid_idx, columns)
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
                         "within_request_auc": within_request_auc(valid_requests, y_valid, prediction),
                         "seen_item_auc": float(roc_auc_score(y_valid[item_seen], prediction[item_seen])),
                         "unseen_item_auc": float(roc_auc_score(y_valid[~item_seen], prediction[~item_seen])),
                         "best_iteration": model.best_iteration, "features": len(columns),
                         "seconds": time.time() - began, "top_gain": [[c, float(g)] for c, g in importance]}
        np.save(args.output / f"valid-{name}.npy", prediction.astype(np.float32))
        print(name, json.dumps({k: v for k, v in results[name].items() if k != "top_gain"}), flush=True)
        print(" top:", [c for c, _ in importance[:15]], flush=True)
        (args.output / "results.json").write_text(json.dumps({
            "cutoff": CUTOFF.isoformat(), "valid_rows": int(len(y_valid)), "fit_rows": int(len(y_fit)),
            "items": len(items), "neighbours": int(index.shape[1]), "cos_pairs": COS_PAIRS,
            "univariate": univariate, "early_stopping_on_holdout": True, "results": results,
            "seconds": time.time() - start}, indent=2) + "\n")
        del x_fit, x_valid, train_set, valid_set, model
        gc.collect()
    np.save(args.output / "valid-row-ids.npy", train["row"].to_numpy()[valid_idx])


if __name__ == "__main__":
    main()
