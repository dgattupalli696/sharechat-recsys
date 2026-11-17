"""Horizon-matched drift screen (local LightGBM).

Train covers 72h and test the following 78h (median 40h ahead), while the
original holdout only measures 0-10h ahead. This screen fits on the first
`--fit-hours` of train and scores the rest by horizon, then measures
permutation importance on near vs far rows to find features that help close to
the training window but hurt further out. `--drop` refits without a list of
features to confirm.
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

from feature_screen import ITEM, LOW_CATEGORICAL, RANKED, scalar_numeric
from request_screen import PARAMS, context

SLICES = [(0, 6), (6, 12), (12, 24), (24, 40)]


def build(args):
    numeric = scalar_numeric(pl.scan_parquet(args.data / "train.parquet").collect_schema())
    frame = pl.scan_parquet(args.data / "train.parquet")
    if args.rows:
        frame = frame.filter(pl.col("request_id").hash(7) % args.rows == 0)
    frame = frame.select("watched", "feature1", "feature3", "request_id", ITEM,
                         *dict.fromkeys([*LOW_CATEGORICAL, *numeric])).collect()
    start = frame["feature1"].min()
    hours = ((frame["feature1"] - start).dt.total_seconds() / 3600).to_numpy()
    is_fit = hours < args.fit_hours
    categories = {c: frame.filter(pl.Series(is_fit))[c].drop_nulls().unique().sort().to_list()
                  for c in LOW_CATEGORICAL}
    wide = json.loads(Path("artifacts/local-t026/results.json").read_text())["wide_features"]
    frame = frame.with_columns(context(wide))
    names = ["feature3", *numeric, *LOW_CATEGORICAL, "rq_size", "rq_items", "rq_item_dup",
             *[f"rq_{k}_{c}" for c in wide for k in ["rank", "dev", "gap"]]]
    parts = []
    for c in names:
        column = pl.col(c)
        if c in categories:
            column = column.cast(pl.Enum(categories[c]), strict=False).to_physical()
        parts.append(frame.select(column.cast(pl.Float32))[c].to_numpy())
    x = np.ascontiguousarray(np.column_stack(parts), dtype=np.float32)
    return x, frame["watched"].to_numpy(), hours - args.fit_hours, is_fit, names, [n for n in names if n in categories]


def sliced(y, p, ahead):
    out = {f"{a}-{b}h": float(roc_auc_score(y[(ahead >= a) & (ahead < b)], p[(ahead >= a) & (ahead < b)]))
           for a, b in SLICES}
    out["all"] = float(roc_auc_score(y, p))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/local-horizon"))
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--fit-hours", type=float, default=32.0)
    parser.add_argument("--rounds", type=int, default=800)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--permute", type=int, default=200, help="Top-gain features to permute (0 = skip)")
    parser.add_argument("--drop", type=Path, help="JSON list of feature-name lists to refit without")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    began = time.time()
    x, y, ahead, is_fit, names, cats = build(args)
    xv, yv, av = x[~is_fit], y[~is_fit], ahead[~is_fit]
    print(f"fit {is_fit.sum()} valid {len(yv)} features {len(names)} in {time.time() - began:.0f}s", flush=True)
    params = {**PARAMS, "learning_rate": .05, "num_threads": args.threads}
    variants = {"base": []}
    if args.drop:
        variants.update({f"drop{i}_{len(d)}": d for i, d in enumerate(json.loads(args.drop.read_text()))})
    results = {}
    for name, dropped in variants.items():
        keep = [i for i, n in enumerate(names) if n not in set(dropped)]
        cols = [names[i] for i in keep]
        model = lgb.train(params, lgb.Dataset(x[is_fit][:, keep], y[is_fit], feature_name=cols,
                                              categorical_feature=[c for c in cats if c in cols]), args.rounds)
        p = model.predict(xv[:, keep])
        results[name] = sliced(yv, p, av)
        print(name, json.dumps({k: round(v, 5) for k, v in results[name].items()}), flush=True)
        if name == "base" and args.permute:
            rng = np.random.default_rng(0)
            near = np.flatnonzero(av < 8)
            far = np.flatnonzero(av >= 24)
            near = rng.choice(near, min(len(near), 120000), replace=False)
            far = rng.choice(far, min(len(far), 120000), replace=False)
            gain = model.feature_importance("gain")
            top = np.argsort(-gain)[:args.permute]
            ref = {k: roc_auc_score(yv[s], p[s]) for k, s in [("near", near), ("far", far)]}
            rows = []
            for j in top:
                drop = {}
                for k, s in [("near", near), ("far", far)]:
                    xs = xv[s][:, keep].copy()
                    xs[:, j] = rng.permutation(xs[:, j])
                    drop[k] = ref[k] - roc_auc_score(yv[s], model.predict(xs))
                rows.append({"feature": cols[j], "gain": float(gain[j]), **drop})
            rows.sort(key=lambda r: r["far"])
            results["permutation"] = rows
            print("most harmful far (negative = permuting helps):", flush=True)
            for r in rows[:25]:
                print(f"  {r['feature']:<28} near {r['near']:+.5f} far {r['far']:+.5f}", flush=True)
        (args.output / "results.json").write_text(json.dumps(
            {"fit_hours": args.fit_hours, "rows": args.rows, "rounds": args.rounds,
             "results": results, "seconds": time.time() - began}, indent=2) + "\n")
        del model
        gc.collect()


if __name__ == "__main__":
    main()
