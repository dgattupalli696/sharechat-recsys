"""T030 local screens: train/test drift and stationary replacements.

adversarial / adversarial_no_dates: LightGBM separating train rows from test rows on
the model inputs (no time column); reports AUC and the most drifting features.

stationarity: on the T026 wide request features and a holdout starting where
feature4 (shifted date) rolls over, like test, compares
  keep_dates           current inputs
  drop_dates           without feature2/feature4 (calendar dates never seen in test)
  stationary           drop_dates, item age instead of upload timestamps feature58/59,
                       per-hour percentiles instead of counters whose level swings
                       between time windows (counters.json, chosen on train+test medians)
  stationary_profiles  plus label-free item/key profiles over train+test
Labels are never used outside the training rows.
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

from feature_screen import CUTOFF, ITEM, LOW_CATEGORICAL, scalar_numeric
from request_screen import PARAMS, context

KEYS = [ITEM, "feature23", "feature20", "feature19"]
DATES = ["feature2", "feature4"]


def adversarial(data, output, threads, rows=400_000, exclude=()):
    numeric = scalar_numeric(pl.scan_parquet(data / "train.parquet").collect_schema())
    columns = [c for c in ["feature3", *numeric, *LOW_CATEGORICAL] if c not in set(exclude)]
    frames = []
    for label, name in enumerate(["train", "test"]):
        frame = pl.scan_parquet(data / f"{name}.parquet").select(columns).collect()
        frames.append(frame.sample(min(rows, frame.height), seed=42).with_columns(pl.lit(label).alias("_is_test")))
    frame = pl.concat(frames)
    del frames
    categories = {c: frame[c].drop_nulls().unique().sort().to_list() for c in LOW_CATEGORICAL if c in columns}
    x = np.column_stack([
        frame.select((pl.col(c).cast(pl.Enum(categories[c])).to_physical() if c in categories
                      else pl.col(c)).cast(pl.Float32))[c].to_numpy() for c in columns
    ]).astype(np.float32)
    y = frame["_is_test"].to_numpy()
    order = np.random.default_rng(0).permutation(len(y))
    fit, valid = order[: int(0.8 * len(y))], order[int(0.8 * len(y)):]
    categorical = [i for i, c in enumerate(columns) if c in categories]
    model = lgb.train({**PARAMS, "learning_rate": 0.1, "num_threads": threads},
                      lgb.Dataset(x[fit], y[fit], feature_name=columns, categorical_feature=categorical),
                      300)
    auc = float(roc_auc_score(y[valid], model.predict(x[valid])))
    gains = sorted(zip(columns, model.feature_importance("gain")), key=lambda p: -p[1])
    total = sum(g for _, g in gains)
    top = [[c, round(float(g / total), 4)] for c, g in gains[:30]]
    single = {}
    for name, _ in top[:30]:
        index = columns.index(name)
        values = np.nan_to_num(x[valid, index], nan=-1e9)
        single[name] = float(max(roc_auc_score(y[valid], values), 1 - roc_auc_score(y[valid], values)))
    report = {"auc": auc, "rows_per_side": rows, "excluded": list(exclude), "top_gain_share": top, "single_feature_auc": single}
    (output / f"adversarial{'-no-' + '-'.join(exclude) if exclude else ''}.json").write_text(json.dumps(report, indent=2) + "\n")
    print("adversarial AUC", round(auc, 4), flush=True)
    for name, share in top[:25]:
        print(f"  {name:12s} gain {share:.3f}  single AUC {single.get(name, 0):.3f}", flush=True)


def transductive_features(data, profiled, counters):
    """Label-free features computed over train+test together, returned for train rows.

    Item profiles: each key's mean of profiled features and the row's deviation.
    Stationary: item age instead of upload timestamps (feature58 s, feature59 ms) and
    per-hour percentiles of counters whose level swings between time windows.
    """
    needed = list(dict.fromkeys(["feature1", "feature58", "feature59", *KEYS, *profiled, *counters]))
    both = pl.concat([
        pl.scan_parquet(data / f"{name}.parquet").select(needed).with_row_index("_row")
        .with_columns(pl.lit(name).alias("_split")).collect()
        for name in ["train", "test"]
    ], how="vertical_relaxed")
    hour = pl.col("feature1").dt.truncate("1h")
    now = pl.col("feature1").dt.epoch("ms").cast(pl.Float64) / 1000
    stationary = [((now - pl.col("feature58")) / 3600).alias("st_age58_h"),
                  ((now - pl.col("feature59") / 1000) / 3600).alias("st_age59_h")]
    stationary += [(pl.col(c).rank("average").over(hour) / pl.col(c).count().over(hour)).alias(f"st_hp_{c}")
                   for c in counters]
    profile = []
    for key in KEYS:
        profile.append((pl.len().over([key, hour]) / pl.len().over(hour)).alias(f"pf_share_{key}"))
        for column in profiled:
            mean = pl.col(column).mean().over(key)
            profile += [mean.alias(f"pf_mean_{key}_{column}"), (pl.col(column) - mean).alias(f"pf_dev_{key}_{column}")]
    exprs = [e.cast(pl.Float32) for e in stationary + profile]
    both = both.with_columns(exprs)
    names = lambda group: [e.meta.output_name() for e in group]
    train = both.filter(pl.col("_split") == "train").sort("_row").select(names(stationary + profile))
    return train, names(stationary), names(profile)


def wide_frame(data, wide, extra=None):
    numeric = scalar_numeric(pl.scan_parquet(data / "train.parquet").collect_schema())
    frame = pl.scan_parquet(data / "train.parquet").select(
        "watched", "feature1", "feature3", "request_id", ITEM,
        *dict.fromkeys([*LOW_CATEGORICAL, *numeric])).collect()
    if extra is not None:
        if extra.height != frame.height:
            raise ValueError("Transductive rows do not align with train rows")
        frame = pl.concat([frame, extra], how="horizontal")
    wide_names = ["rq_size", "rq_items", "rq_item_dup"] + [f"rq_{k}_{c}" for c in wide for k in ["rank", "dev", "gap"]]
    return frame.with_columns(context(wide)), ["feature3", *numeric, *LOW_CATEGORICAL, *wide_names]


def fit_variants(holder, cutoff, variants, threads, output, name):
    """holder: one-item list owning the frame, so it is freed once matrices are built."""
    started = time.time()
    frame = holder.pop()
    is_valid = (frame["feature1"] >= cutoff).to_numpy()
    categories = {c: frame.filter(pl.Series(~is_valid))[c].drop_nulls().unique().sort().to_list()
                  for c in LOW_CATEGORICAL}
    union = list(dict.fromkeys(c for columns in variants.values() for c in columns))
    y = frame["watched"].to_numpy()
    x_fit = np.empty((int((~is_valid).sum()), len(union)), dtype=np.float32)
    x_valid = np.empty((int(is_valid.sum()), len(union)), dtype=np.float32)
    for i, c in enumerate(union):
        column = pl.col(c)
        if c in categories:
            column = column.cast(pl.Enum(categories[c]), strict=False).to_physical()
        values = frame.select(column.cast(pl.Float32))[c].to_numpy()
        x_fit[:, i], x_valid[:, i] = values[~is_valid], values[is_valid]
    del frame
    gc.collect()
    print(f"{name}: cutoff {cutoff}, fit {len(x_fit):,}, holdout {len(x_valid):,}, columns {len(union)}", flush=True)
    path = output / f"{name}.json"
    results = {}
    for label, columns in variants.items():
        began = time.time()
        index = [union.index(c) for c in columns]
        categorical = [i for i, c in enumerate(columns) if c in categories]
        train_set = lgb.Dataset(x_fit[:, index], y[~is_valid], feature_name=columns,
                                categorical_feature=categorical, free_raw_data=True)
        valid_x = x_valid[:, index]
        valid_set = lgb.Dataset(valid_x, y[is_valid], reference=train_set)
        model = lgb.train({**PARAMS, "learning_rate": .05, "num_threads": threads}, train_set, 4000,
                          valid_sets=[valid_set], valid_names=["holdout"],
                          callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(250)])
        auc = float(roc_auc_score(y[is_valid], model.predict(valid_x, num_iteration=model.best_iteration)))
        gains = sorted(zip(columns, model.feature_importance("gain")), key=lambda p: -p[1])
        total = sum(g for _, g in gains)
        results[label] = {"holdout_auc": auc, "best_iteration": model.best_iteration,
                          "features": len(columns), "seconds": time.time() - began,
                          "top_gain": [[c, float(g / total)] for c, g in gains[:40]]}
        print(label, json.dumps({k: v for k, v in results[label].items() if k != "top_gain"}), flush=True)
        path.write_text(json.dumps({"cutoff": str(cutoff), "holdout_rows": int(len(valid_x)),
                                    "variants": {k: v for k, v in variants.items()},
                                    "results": results, "seconds": time.time() - started}, indent=2) + "\n")
        del train_set, valid_set, valid_x, model
        gc.collect()
    return results


def stationarity(data, output, threads, wide_from, counters_from, profile_count, cutoff):
    wide = json.loads(wide_from.read_text())["wide_features"]
    counters = json.loads(counters_from.read_text())
    extra, stationary, profile = transductive_features(data, wide[:profile_count], counters)
    holder = [None]
    holder[0], columns = wide_frame(data, wide, extra)
    del extra
    replaced = set(DATES) | {"feature58", "feature59"} | set(counters)
    stable = [c for c in columns if c not in replaced] + stationary
    fit_variants(holder, cutoff, {
        "keep_dates": columns,
        "drop_dates": [c for c in columns if c not in DATES],
        "stationary": stable,
        "stationary_profiles": stable + profile,
    }, threads, output, f"stationarity-{cutoff:%m%dT%H%M}")


def main():
    from datetime import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/local-t030"))
    parser.add_argument("--stages", default="adversarial,adversarial_no_dates,stationarity")
    parser.add_argument("--wide-from", type=Path, default=Path("artifacts/local-t026/results.json"))
    parser.add_argument("--counters-from", type=Path, default=Path("artifacts/local-t030/counters.json"))
    parser.add_argument("--profile-count", type=int, default=12)
    parser.add_argument("--cutoff", default="2026-10-27T18:00:00",
                        help="Holdout start; 18:00 is where feature4 (shifted date) rolls over, like test")
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cutoff = datetime.fromisoformat(args.cutoff)
    for stage in args.stages.split(","):
        if stage == "adversarial":
            adversarial(args.data, args.output, args.threads)
        elif stage == "adversarial_no_dates":
            adversarial(args.data, args.output, args.threads, exclude=DATES)
        elif stage == "stationarity":
            stationarity(args.data, args.output, args.threads, args.wide_from, args.counters_from,
                         args.profile_count, cutoff)
        else:
            raise ValueError(f"Unknown stage: {stage}")
        gc.collect()


if __name__ == "__main__":
    main()
