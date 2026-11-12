import argparse
import gc
import json
from pathlib import Path

from catboost import CatBoostClassifier
from catboost.utils import get_gpu_device_count
import numpy as np
import polars as pl
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parent
ON_KAGGLE = Path("/kaggle/input").is_dir()
TARGET = "watched"
TIME = "feature1"


def features(frame):
    expressions = []
    for name, dtype in frame.schema.items():
        if name in {TARGET, "request_id", "id"}:
            continue
        column = pl.col(name)
        if isinstance(dtype, pl.Datetime):
            expressions.extend(
                [
                    column.dt.hour().cast(pl.Float32).alias(f"{name}_hour"),
                    column.dt.weekday().cast(pl.Float32).alias(f"{name}_weekday"),
                ]
            )
        elif isinstance(dtype, pl.List):
            expressions.extend(
                [
                    column.list.len().cast(pl.Float32).alias(f"{name}_length"),
                    column.list.mean().cast(pl.Float32).alias(f"{name}_mean"),
                    column.list.std().cast(pl.Float32).alias(f"{name}_std"),
                    column.list.min().cast(pl.Float32).alias(f"{name}_min"),
                    column.list.max().cast(pl.Float32).alias(f"{name}_max"),
                ]
            )
        elif dtype == pl.String:
            expressions.append(column.fill_null("__MISSING__").alias(name))
        elif dtype.is_numeric():
            expressions.append(column.cast(pl.Float32).alias(name))
        else:
            raise TypeError(f"Unsupported feature {name}: {dtype}")
    return frame.select(expressions).with_columns(
        pl.col(pl.Float32).fill_nan(None)
    )


def load_sample(path, maximum_rows, seed):
    parquet = pq.ParquetFile(path)
    total = parquet.metadata.num_rows
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(total, min(maximum_rows, total), replace=False))
    frames = []
    offset = 0
    for batch in parquet.iter_batches(batch_size=8192):
        end = offset + batch.num_rows
        selected = indices[(indices >= offset) & (indices < end)] - offset
        if len(selected):
            raw = pl.from_arrow(batch.take(selected))
            frames.append(
                features(raw).with_columns(raw[TARGET], raw[TIME].alias("_time"))
            )
        offset = end
    return pl.concat(frames)


def model(iterations, threads, task_type):
    return CatBoostClassifier(
        iterations=iterations,
        depth=6,
        learning_rate=0.06,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        thread_count=threads,
        task_type=task_type,
        max_ctr_complexity=1,
        allow_writing_files=False,
        verbose=50,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1_600_000 if ON_KAGGLE else 150_000)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--task-type", choices=["CPU", "GPU"], default="GPU" if ON_KAGGLE else "CPU"
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/kaggle/working") if ON_KAGGLE else ROOT / "artifacts",
    )
    args = parser.parse_args()
    if min(args.rows, args.iterations, args.threads) <= 0:
        parser.error("rows, iterations, and threads must be positive")

    if args.task_type == "GPU" and get_gpu_device_count() == 0:
        raise RuntimeError("GPU training requested, but no CUDA GPU is available")
    data = args.data_dir
    if data is None and ON_KAGGLE:
        candidates = [
            path.parent
            for path in Path("/kaggle/input").rglob("train.parquet")
            if (path.parent / "test.parquet").is_file()
            and (path.parent / "sample_submission.csv").is_file()
        ]
        if len(candidates) != 1:
            raise RuntimeError(f"Expected one ShareChat dataset, found {candidates}")
        data = candidates[0]
    elif data is None:
        data = ROOT / "data"
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    print(f"Compute: {args.task_type}; input: {data}; output: {output}", flush=True)
    sample = load_sample(data / "train.parquet", args.rows, 42).sort("_time")
    if sample.height < 2:
        raise ValueError("Chronological validation requires at least two rows")
    if sample["_time"].null_count():
        raise ValueError("Missing timestamps prevent chronological validation")
    cutoff = sample["_time"][int(sample.height * 0.8)]
    train = sample.filter(pl.col("_time") < cutoff)
    valid = sample.filter(pl.col("_time") >= cutoff)
    if train.is_empty() or valid.is_empty():
        raise ValueError(
            f"Timestamp cutoff {cutoff} cannot produce nonempty chronological splits"
        )
    for name, part in [("train", train), ("validation", valid)]:
        if part[TARGET].n_unique() != 2:
            raise ValueError(f"{name} must contain both target classes")
    columns = [name for name in sample.columns if name not in {TARGET, "_time"}]
    categories = [name for name in columns if sample.schema[name] == pl.String]
    x_train = train.select(columns).to_pandas()
    x_valid = valid.select(columns).to_pandas()
    print(
        f"Training: {train.height:,}; validation: {valid.height:,}; "
        f"features: {len(columns)}; cutoff: {cutoff}",
        flush=True,
    )
    estimator = model(args.iterations, args.threads, args.task_type)
    estimator.fit(
        x_train,
        train[TARGET].to_numpy(),
        cat_features=categories,
        eval_set=(x_valid, valid[TARGET].to_numpy()),
        early_stopping_rounds=60,
    )
    auc = roc_auc_score(
        valid[TARGET].to_numpy(), estimator.predict_proba(x_valid)[:, 1]
    )
    best_iterations = estimator.tree_count_
    metrics = {
        "validation_auc": auc,
        "validation": "chronological 80/20 split; timestamp ties kept together",
        "cutoff": str(cutoff),
        "sample_rows": sample.height,
        "training_rows": train.height,
        "validation_rows": valid.height,
        "iterations": best_iterations,
        "task_type": args.task_type,
        "features": columns,
        "categorical_features": categories,
    }
    estimator.save_model(str(output / "validation.cbm"))
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Validation ROC AUC: {auc:.6f}", flush=True)
    del estimator, x_train, x_valid, train, valid
    gc.collect()

    print("Refitting on the full sample", flush=True)
    estimator = model(best_iterations, args.threads, args.task_type)
    estimator.fit(
        sample.select(columns).to_pandas(),
        sample[TARGET].to_numpy(),
        cat_features=categories,
    )
    estimator.save_model(str(output / "baseline.cbm"))
    del sample
    gc.collect()

    submission = pl.read_csv(data / "sample_submission.csv")
    if submission.columns != ["id", TARGET]:
        raise ValueError(f"Unexpected submission schema: {submission.columns}")
    test = pq.ParquetFile(data / "test.parquet")
    if test.metadata.num_rows != submission.height:
        raise ValueError("Test and submission row counts differ")
    predictions = []
    offset = 0
    for batch in test.iter_batches(batch_size=8192):
        raw = pl.from_arrow(batch)
        if "id" not in raw.columns:
            raise ValueError("Test data has no id column for submission alignment")
        expected = submission["id"].slice(offset, raw.height)
        if raw["id"].to_list() != expected.to_list():
            raise ValueError("Test ids do not match sample submission order")
        predictions.append(
            estimator.predict_proba(features(raw).select(columns).to_pandas())[:, 1]
        )
        offset += raw.height
    probabilities = np.concatenate(predictions)
    if not np.isfinite(probabilities).all() or not (
        (probabilities >= 0) & (probabilities <= 1)
    ).all():
        raise ValueError("Invalid predicted probabilities")
    submission.with_columns(pl.Series(TARGET, probabilities)).write_csv(
        output / "submission.csv"
    )
    print(f"Saved {submission.height:,} predictions to {output / 'submission.csv'}")


if __name__ == "__main__":
    main()
