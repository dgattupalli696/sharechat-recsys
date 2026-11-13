import argparse
import json
from pathlib import Path
import resource
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd


class ArraySequence(lgb.Sequence):
    batch_size = 8192

    def __init__(self, path):
        self.values = np.load(path, mmap_mode="r")

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        return np.asarray(self.values[index], dtype=np.float64)


def train_arrays(directory, output, iterations, threads, *, fixed_budget=False):
    metadata = json.loads((directory / "metadata.json").read_text())
    labels = np.load(directory / "labels.npz")
    train_sequence = ArraySequence(directory / "train.npy")
    valid_sequence = ArraySequence(directory / "valid.npy")
    params = {
        "objective": "binary", "metric": "auc", "learning_rate": 0.03,
        "num_leaves": 31, "max_depth": -1, "min_data_in_leaf": 100,
        "feature_fraction": 0.8, "lambda_l2": 1, "seed": 42,
        "num_threads": threads, "verbosity": -1, "device_type": "cpu",
        "force_col_wise": True, "histogram_pool_size": 512,
    }
    training = lgb.Dataset(
        train_sequence, label=labels["train"], feature_name=metadata["columns"],
        categorical_feature=metadata["categories"], params=params,
    )
    if fixed_budget:
        booster = lgb.train(params, training, num_boost_round=iterations)
    else:
        validation = lgb.Dataset(
            valid_sequence, label=labels["valid"], reference=training,
            feature_name=metadata["columns"], categorical_feature=metadata["categories"],
        )
        booster = lgb.train(
            params, training, num_boost_round=iterations,
            valid_sets=[validation], valid_names=["validation"],
            callbacks=[lgb.early_stopping(60), lgb.log_evaluation(50)],
        )
    selected_iteration = iterations if fixed_budget else booster.best_iteration
    probabilities = np.empty(len(valid_sequence), dtype=np.float64)
    for start in range(0, len(valid_sequence), valid_sequence.batch_size):
        end = min(start + valid_sequence.batch_size, len(valid_sequence))
        probabilities[start:end] = booster.predict(
            valid_sequence[start:end], num_iteration=selected_iteration,
            num_threads=threads,
        )
    np.savez_compressed(
        output / "validation_predictions.npz", y_true=labels["valid"],
        probabilities=probabilities, row_ids=labels["row_ids"],
    )
    booster.save_model(str(output / "model.txt"))
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    (output / "worker-results.json").write_text(json.dumps({
        "best_iteration": selected_iteration, "library_version": lgb.__version__,
        "selection_mode": "fixed_budget" if fixed_budget else "validation",
        "backend": "CPU isolated disk-backed process", "parameters": params,
        "peak_rss_bytes": peak if sys.platform == "darwin" else peak * 1024,
    }, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--input-dir", type=Path)
    inputs.add_argument("--array-dir", type=Path)
    inputs.add_argument("--predict-array", type=Path)
    parser.add_argument("--model-file", type=Path)
    parser.add_argument("--feature-columns", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--fixed-budget", action="store_true")
    args = parser.parse_args()
    if args.predict_array is not None:
        if args.model_file is None or args.feature_columns is None:
            parser.error("--model-file and --feature-columns are required for prediction")
        booster = lgb.Booster(model_file=str(args.model_file))
        columns = json.loads(args.feature_columns.read_text())
        matrix = np.load(args.predict_array, mmap_mode="r")
        if matrix.ndim != 2 or matrix.shape[1] != booster.num_feature() or columns != booster.feature_name():
            raise ValueError("Prediction features differ from the saved LightGBM model")
        probabilities = np.empty(len(matrix), dtype=np.float64)
        for start in range(0, len(matrix), 8192):
            probabilities[start:start + 8192] = booster.predict(
                np.asarray(matrix[start:start + 8192], dtype=np.float64),
                num_threads=args.threads,
            )
        if not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
            raise ValueError("Invalid saved-model LightGBM predictions")
        np.save(args.output_dir / "probabilities.npy", probabilities)
        return
    if args.array_dir is not None:
        train_arrays(
            args.array_dir, args.output_dir, args.iterations, args.threads,
            fixed_budget=args.fixed_budget,
        )
        return
    train = pd.read_parquet(args.input_dir / "train.parquet")
    valid = pd.read_parquet(args.input_dir / "valid.parquet")
    labels = np.load(args.input_dir / "labels.npz")
    categories = json.loads((args.input_dir / "categories.json").read_text())
    for column in categories:
        dtype = pd.CategoricalDtype(categories=pd.unique(train[column]))
        train[column] = train[column].astype(dtype)
        valid[column] = valid[column].where(valid[column].isin(dtype.categories), None).astype(dtype)
    estimator = lgb.LGBMClassifier(
        objective="binary", metric="auc", n_estimators=args.iterations,
        learning_rate=0.03, num_leaves=31, max_depth=-1,
        min_child_samples=100, colsample_bytree=0.8,
        reg_lambda=1, random_state=42, n_jobs=4, verbosity=-1, device_type="cpu",
    )
    if args.fixed_budget:
        estimator.fit(train, labels["train"])
    else:
        estimator.fit(
            train, labels["train"], eval_set=[(valid, labels["valid"])],
            callbacks=[lgb.early_stopping(60), lgb.log_evaluation(50)],
        )
    probabilities = estimator.predict_proba(valid)[:, 1]
    np.savez_compressed(
        args.output_dir / "validation_predictions.npz",
        y_true=labels["valid"], probabilities=probabilities,
    )
    estimator.booster_.save_model(str(args.output_dir / "model.txt"))
    (args.output_dir / "worker-results.json").write_text(json.dumps({
        "best_iteration": args.iterations if args.fixed_budget else estimator.best_iteration_,
        "library_version": lgb.__version__,
        "backend": "CPU isolated process",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
