import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--fixed-budget", action="store_true")
    args = parser.parse_args()
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
