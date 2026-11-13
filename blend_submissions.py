"""Candidate submissions from full-data refits, with blend weights chosen on the original holdout.

Each component has holdout predictions (from the fit-period run) and one or more full-refit
test predictions (seeds are averaged). Blends are rank averages; weights are searched on a
simplex grid against the 307,528-row holdout and then applied unchanged to test ranks.
"""
import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

HOLDOUT = {
    "tabm": "artifacts/kaggle-t027/t027/ensemble-32/validation_predictions.npz",
    "xgb63": "artifacts/kaggle-t036/t036/leaves63_eta02-holdout.npz",
    "xgb511": "artifacts/kaggle-t036/t036/leaves511_eta02-holdout.npz",
    "xgbd12": "artifacts/kaggle-t036/t036/depth12_node-holdout.npz",
}
TEST = {
    "tabm": ["artifacts/kaggle-t037/refit/test-component-predictions.npz:tabm32",
             "artifacts/kaggle-t037/refit/test-component-predictions.npz:tabm32-seed43",
             "artifacts/kaggle-t037/refit/test-component-predictions.npz:tabm32-seed44"],
    "xgb63": ["artifacts/kaggle-t038/t038/leaves63_eta02-s42-test.npz:probabilities",
              "artifacts/kaggle-t038/t038/leaves63_eta02-s43-test.npz:probabilities"],
    "xgb511": ["artifacts/kaggle-t038/t038/leaves511_eta02-s42-test.npz:probabilities",
               "artifacts/kaggle-t038/t038/leaves511_eta02-s43-test.npz:probabilities"],
    "xgbd12": ["artifacts/kaggle-t038/t038/depth12_node-s42-test.npz:probabilities",
               "artifacts/kaggle-t038/t038/depth12_node-s43-test.npz:probabilities"],
}
REFERENCE = "artifacts/kaggle-t025/refit/submission-full-tabm.csv"


def ranks(values):
    return rankdata(values) / len(values)


def holdout_ranks(names):
    frame = None
    for name in names:
        z = np.load(HOLDOUT[name])
        part = pl.DataFrame({"row": z["row_ids"].astype(np.int64), "y": z["y_true"].astype(np.int8),
                             name: z["probabilities"].astype(np.float64)})
        frame = part if frame is None else frame.join(part.drop("y"), on="row", how="inner")
    if frame is None or len(frame) != 307_528:
        raise ValueError("Holdout predictions do not cover the original holdout")
    return frame["y"].to_numpy(), {n: ranks(frame[n].to_numpy()) for n in names}


def search(y, components, step):
    names = list(components)
    grid = np.round(np.arange(0, 1 + step / 2, step), 6)
    best = (-1.0, None)
    for weights in product(grid, repeat=len(names) - 1):
        last = 1 - sum(weights)
        if last < -1e-9:
            continue
        w = [*weights, max(last, 0.0)]
        auc = roc_auc_score(y, sum(wi * components[n] for wi, n in zip(w, names)))
        if auc > best[0]:
            best = (auc, dict(zip(names, map(float, w))))
    return best


def test_ranks(name):
    seeds = []
    for spec in TEST[name]:
        path, key = spec.split(":")
        seeds.append(np.load(path)[key].astype(np.float64))
    return ranks(np.mean(seeds, axis=0)), len(seeds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/candidates"))
    parser.add_argument("--template", type=Path, default=Path("data/sample_submission.csv"))
    parser.add_argument("--step", type=float, default=0.05)
    args = parser.parse_args()
    available = [n for n in HOLDOUT if Path(HOLDOUT[n]).is_file()
                 and all(Path(s.split(":")[0]).is_file() for s in TEST[n])]
    print("components:", available, flush=True)
    y, holdout = holdout_ranks(available)
    template = pl.read_csv(args.template)
    reference = pl.read_csv(REFERENCE)
    if not template["id"].equals(reference["id"]):
        raise ValueError("Reference submission order differs from the template")
    tests = {n: test_ranks(n) for n in available}
    trees = [n for n in available if n.startswith("xgb")]
    candidates = {n: {n: 1.0} for n in available}
    for group in [["tabm", "xgbd12"], ["tabm", "xgb63", "xgbd12"], ["tabm", *trees], trees]:
        if len(group) > 1 and all(n in available for n in group):
            auc, weights = search(y, {n: holdout[n] for n in group}, args.step)
            candidates["blend-" + "-".join(group)] = weights
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, weights in candidates.items():
        auc = roc_auc_score(y, sum(w * holdout[n] for n, w in weights.items()))
        values = sum(w * tests[n][0] for n, w in weights.items())
        if len(values) != len(template) or not np.isfinite(values).all():
            raise ValueError(f"Invalid test predictions: {name}")
        path = args.output / f"submission-{name}.csv"
        template.with_columns(pl.Series("watched", values.astype(np.float64))).write_csv(path)
        summary[name] = {
            "file": str(path), "weights": weights, "holdout_auc": float(auc),
            "test_seeds": {n: tests[n][1] for n in weights},
            "spearman_vs_s007": float(spearmanr(values, reference["watched"].to_numpy()).statistic),
        }
        print(name, json.dumps({k: summary[name][k] for k in ["weights", "holdout_auc", "spearman_vs_s007"]}),
              flush=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
