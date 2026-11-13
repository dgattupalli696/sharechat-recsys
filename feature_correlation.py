import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from train import features


PRIMARY = datetime.fromisoformat("2026-10-28T08:06:52.397000")
EARLIEST = datetime.fromisoformat("2026-10-27T08:06:52.397000")


class Moments:
    def __init__(self, width):
        self.count = np.zeros(width, dtype=np.int64)
        self.mean = np.zeros(width, dtype=np.float64)
        self.m2 = np.zeros(width, dtype=np.float64)

    def update(self, array):
        finite = np.isfinite(array)
        count = finite.sum(axis=0)
        safe = np.where(finite, array, 0).astype(np.float64)
        mean = np.divide(safe.sum(axis=0), count, out=np.zeros_like(self.mean), where=count > 0)
        m2 = np.square(np.where(finite, safe - mean, 0)).sum(axis=0)
        total = self.count + count
        delta = mean - self.mean
        fraction = np.divide(count, total, out=np.zeros_like(mean), where=total > 0)
        self.mean += delta * fraction
        self.m2 += m2 + delta ** 2 * self.count * fraction
        self.count = total


def target_correlations(negative, positive):
    total = negative.count + positive.count
    cross = np.divide(
        negative.count.astype(float) * positive.count, total,
        out=np.zeros_like(negative.mean), where=total > 0,
    )
    delta = positive.mean - negative.mean
    variance = negative.m2 + positive.m2 + delta ** 2 * cross
    denominator = np.sqrt(variance * cross)
    return np.divide(
        delta * cross, denominator,
        out=np.full_like(delta, np.nan), where=denominator > 0,
    )


def pairwise_correlations(chunks, moments):
    width = len(moments.mean)
    scale = np.sqrt(np.divide(
        moments.m2, moments.count, out=np.zeros(width), where=moments.count > 0,
    ))
    scale[scale == 0] = 1
    products, sums, squares, counts = [np.zeros((width, width), dtype=np.float64) for _ in range(4)]
    for array in chunks:
        finite = np.isfinite(array)
        mask = finite.astype(np.float32)
        centered = np.where(finite, (array.astype(np.float64) - moments.mean) / scale, 0).astype(np.float32)
        products += centered.T @ centered
        sums += centered.T @ mask
        squares += np.square(centered).T @ mask
        counts += mask.T @ mask
    safe_count = np.maximum(counts, 1)
    covariance = products - sums * sums.T / safe_count
    variance = np.maximum(squares - sums ** 2 / safe_count, 0)
    denominator = np.sqrt(variance * variance.T)
    correlation = np.divide(
        covariance, denominator, out=np.full_like(covariance, np.nan),
        where=(denominator > 0) & (counts >= 3),
    )
    return np.clip(correlation, -1, 1), counts.astype(np.int64)


def numeric_chunks(path, names=None):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=8192):
        raw = pl.from_arrow(batch).filter(pl.col("feature1") < PRIMARY)
        if raw.is_empty():
            continue
        frame = features(raw, expand_embeddings=True)
        selected = names or [name for name, dtype in frame.schema.items() if dtype.is_numeric()]
        yield selected, np.array(frame.select(selected).to_numpy(), dtype=np.float32), raw


def categorical_analysis(path):
    schema = pl.scan_parquet(path).collect_schema()
    fields = [name for name, dtype in schema.items() if dtype == pl.String and name != "request_id"]
    metadata = pl.scan_parquet(path).select("feature1", "watched").filter(
        pl.col("feature1") < PRIMARY,
    ).collect()
    times = metadata["feature1"].to_numpy()
    early = times < np.datetime64(EARLIEST)
    early_times = np.sort(times[early])
    cutoff = early_times[int(.8 * len(early_times))]
    fit = early & (times < cutoff)
    score = early & ~fit
    labels = metadata["watched"].to_numpy()
    prior = float(labels[fit].mean())
    encoded, report = [], []
    for name in fields:
        values = pl.scan_parquet(path).filter(pl.col("feature1") < PRIMARY).select(name).collect()[name]
        codes, unique = pd.factorize(values.to_pandas(), sort=False, use_na_sentinel=True)
        codes = (codes + 1).astype(np.int32)
        encoded.append(codes)
        cardinality = len(unique) + 1
        count = np.bincount(codes[fit], minlength=cardinality)
        positive = np.bincount(codes[fit], weights=labels[fit], minlength=cardinality)
        rates = (positive + 20 * prior) / (count + 20)
        predictions = rates[codes[score]]
        report.append({
            "feature": name, "primary_training_unique_nonmissing": len(unique),
            "inner_training_rows": int(fit.sum()), "inner_validation_rows": int(score.sum()),
            "inner_temporal_auc": float(roc_auc_score(labels[score], predictions)),
            "inner_unknown_fraction": float((count[codes[score]] == 0).mean()),
            "missing_fraction": values.null_count() / len(values),
        })
    dependencies = []
    for i, left in enumerate(encoded):
        for j in range(i + 1, len(encoded)):
            right = encoded[j]
            radix = int(right.max()) + 1
            keys, count = np.unique(left.astype(np.int64) * radix + right, return_counts=True)
            a, b = keys // radix, keys % radix
            starts = np.r_[0, np.flatnonzero(a[1:] != a[:-1]) + 1]
            errors_forward = 1 - np.maximum.reduceat(count, starts).sum() / len(left)
            order = np.argsort(b, kind="stable")
            sorted_b = b[order]
            starts = np.r_[0, np.flatnonzero(sorted_b[1:] != sorted_b[:-1]) + 1]
            errors_reverse = 1 - np.maximum.reduceat(count[order], starts).sum() / len(left)
            if min(errors_forward, errors_reverse) <= .001:
                dependencies.append({
                    "first": fields[i], "second": fields[j], "unique_pairs": len(keys),
                    "first_to_second_error_fraction": float(errors_forward),
                    "second_to_first_error_fraction": float(errors_reverse),
                    "near_bijection": bool(max(errors_forward, errors_reverse) <= .001),
                })
    return {
        "target_associations": report, "functional_dependencies": dependencies,
        "inner_cutoff": str(cutoff), "outer_holdout_labels_used": False,
        "interpretation": "Category-rate AUC uses a forward split entirely inside the earliest training prefix, smoothing strength 20; not a multivariate feature-importance measure.",
    }


def run(path, output, threads=4):
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    names = next(numeric_chunks(path))[0]
    moments, negative, positive = [Moments(len(names)) for _ in range(3)]
    digests = [hashlib.sha256() for _ in names]
    missing, infinities = np.zeros(len(names), dtype=np.int64), np.zeros(len(names), dtype=np.int64)
    missing_negative = np.zeros(len(names), dtype=np.int64)
    missing_positive = np.zeros(len(names), dtype=np.int64)
    rows, target_rows, classes = 0, 0, np.zeros(2, dtype=np.int64)
    for _, array, raw in numeric_chunks(path, names):
        moments.update(array)
        missing += np.isnan(array).sum(axis=0)
        infinities += np.isinf(array).sum(axis=0)
        early = raw["feature1"].to_numpy() < np.datetime64(EARLIEST)
        target = raw["watched"].to_numpy()
        for label, stats in [(0, negative), (1, positive)]:
            mask = early & (target == label)
            stats.update(array[mask])
            classes[label] += mask.sum()
            absent = (~np.isfinite(array[mask])).sum(axis=0)
            if label:
                missing_positive += absent
            else:
                missing_negative += absent
        canonical = array.copy()
        canonical[~np.isfinite(canonical)] = np.nan
        for digest, values in zip(digests, np.ascontiguousarray(canonical.T)):
            digest.update(values.tobytes())
        rows += len(array)
        target_rows += int(early.sum())
    print(f"Numeric moments: {rows:,} full training rows, {len(names)} features; target prefix {target_rows:,}", flush=True)
    with threadpool_limits(limits=threads):
        correlation, pair_counts = pairwise_correlations(
            (array for _, array, _ in numeric_chunks(path, names)), moments,
        )
    np.savez_compressed(
        output / "numeric-correlations.npz", features=np.array(names),
        correlation=correlation, observed_pair_counts=pair_counts,
    )
    print("All pairwise-observed numeric correlations complete", flush=True)
    target_r = target_correlations(negative, positive)
    missing_zero, missing_one = Moments(len(names)), Moments(len(names))
    for counts, total, stats in [
        (missing_negative, classes[0], missing_zero), (missing_positive, classes[1], missing_one),
    ]:
        stats.count[:] = total
        stats.mean[:] = counts / total
        stats.m2[:] = counts * (1 - counts / total)
    missing_r = target_correlations(missing_zero, missing_one)
    numeric = []
    duplicates = {}
    for i, name in enumerate(names):
        numeric.append({
            "feature": name, "observed_rows": int(moments.count[i]),
            "missing_fraction": float(missing[i] / rows),
            "infinite_fraction": float(infinities[i] / rows),
            "mean": float(moments.mean[i]) if moments.count[i] else None,
            "standard_deviation": float(np.sqrt(moments.m2[i] / moments.count[i])) if moments.count[i] else None,
            "target_pearson_earliest_train": float(target_r[i]) if np.isfinite(target_r[i]) else None,
            "missingness_target_pearson_earliest_train": float(missing_r[i]) if np.isfinite(missing_r[i]) else None,
        })
        if moments.count[i] and moments.m2[i] > 0:
            duplicates.setdefault(digests[i].hexdigest(), []).append(name)
    upper = np.triu(np.ones(correlation.shape, dtype=bool), k=1)
    indices = np.argwhere(upper & (np.abs(correlation) >= .98))
    pairs = sorted([
        {"first": names[i], "second": names[j], "pearson": float(correlation[i, j]),
         "observed_pair_rows": int(pair_counts[i, j])}
        for i, j in indices
    ], key=lambda row: -abs(row["pearson"]))
    categorical = categorical_analysis(path)
    report = {
        "status": "COMPLETE", "training_rows": rows,
        "numeric_feature_count": len(names), "categorical_feature_count": len(categorical["target_associations"]),
        "numeric_columns": numeric,
        "constant_or_all_missing": [names[i] for i in range(len(names)) if moments.m2[i] == 0],
        "exact_numeric_duplicates_nonfinite_as_missing": [group for group in duplicates.values() if len(group) > 1],
        "strong_numeric_pairs": pairs, "categorical": categorical,
        "target_association_rows": target_rows, "target_association_cutoff": EARLIEST.isoformat(),
        "correlation_cutoff": PRIMARY.isoformat(), "threads": threads,
        "correlation_definition": "Pearson on pairwise finite observations; standardized float32 block products accumulated in float64; no row subsampling or imputation",
        "competition_submission": False, "running_model_experiment_modified": False,
        "seconds": time.monotonic() - started,
    }
    (output / "correlation-results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    pl.DataFrame(numeric).write_csv(output / "numeric-feature-summary.csv")
    pl.DataFrame(categorical["target_associations"]).write_csv(output / "categorical-target-associations.csv")
    print(f"Audit complete in {report['seconds']:.1f}s: {len(pairs)} strong numeric pairs", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/train.parquet"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("Thread count must be positive")
    run(args.input, args.output, args.threads)


if __name__ == "__main__":
    main()
