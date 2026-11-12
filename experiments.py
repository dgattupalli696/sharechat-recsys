import argparse
import gc
import json
from pathlib import Path
import time

import polars as pl
from sklearn.metrics import roc_auc_score

from train import ROOT, TARGET, load_sample, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=150_000)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/local-v1")
    args = parser.parse_args()
    if min(args.rows, args.iterations, args.threads) <= 0:
        parser.error("rows, iterations, and threads must be positive")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    sample = load_sample(
        ROOT / "data/train.parquet", args.rows, 42, expand_embeddings=True
    ).sort("_time")
    if sample.height < 2 or sample["_time"].null_count():
        raise ValueError("Validation requires nonmissing timestamps and multiple rows")
    cutoff = sample["_time"][int(sample.height * 0.8)]
    train = sample.filter(pl.col("_time") < cutoff)
    valid = sample.filter(pl.col("_time") >= cutoff)
    if train[TARGET].n_unique() != 2 or valid[TARGET].n_unique() != 2:
        raise ValueError("Both chronological partitions must contain both classes")
    sample.write_parquet(output / "sample.parquet")
    preparation_seconds = time.monotonic() - started
    print(
        f"Prepared {sample.height:,} rows in {preparation_seconds:.1f}s; "
        f"train={train.height:,}; valid={valid.height:,}; cutoff={cutoff}",
        flush=True,
    )
    results = []
    for name, expanded in [("summary_baseline", False), ("expanded_embeddings", True)]:
        columns = [
            column
            for column in sample.columns
            if column not in {TARGET, "_time"} and (expanded or "_dim_" not in column)
        ]
        categories = [column for column in columns if sample.schema[column] == pl.String]
        x_train = train.select(columns).to_pandas()
        x_valid = valid.select(columns).to_pandas()
        estimator = model(args.iterations, args.threads, "CPU")
        print(f"Experiment: {name}; features={len(columns)}", flush=True)
        started = time.monotonic()
        estimator.fit(
            x_train,
            train[TARGET].to_numpy(),
            cat_features=categories,
            eval_set=(x_valid, valid[TARGET].to_numpy()),
            early_stopping_rounds=60,
        )
        training_seconds = time.monotonic() - started
        probabilities = estimator.predict_proba(x_valid)[:, 1]
        result = {
            "experiment": name,
            "validation_auc": roc_auc_score(valid[TARGET].to_numpy(), probabilities),
            "training_seconds": training_seconds,
            "iterations": estimator.tree_count_,
            "features": len(columns),
        }
        estimator.save_model(str(output / f"{name}.cbm"))
        results.append(result)
        report = {
            "sample_rows": sample.height,
            "seed": 42,
            "training_rows": train.height,
            "validation_rows": valid.height,
            "cutoff": str(cutoff),
            "preparation_seconds": preparation_seconds,
            "task_type": "CPU",
            "threads": args.threads,
            "max_iterations": args.iterations,
            "results": results,
        }
        (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        del estimator, x_train, x_valid
        gc.collect()


if __name__ == "__main__":
    main()
