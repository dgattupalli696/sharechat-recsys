import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiment_analysis import load_candidates, paired_group_bootstrap
from relational_experiment import METHODS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts/kaggle-t019/relational")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    all_results = {}
    for fold, counts in [("primary", (1230110, 307528)), ("earlier", (710330, 299969))]:
        directory = args.artifacts / fold
        protocol = json.loads((directory / "relational-protocol.json").read_text())
        original = json.loads((ROOT / "artifacts/kaggle-t018" / fold / "protocol.json").read_text())
        assert (protocol["training_rows"], protocol["validation_rows"]) == counts
        assert protocol["validation_labels_sha256"] == original["validation_labels_sha256"]
        assert protocol["validation_identities_sha256"] == original["validation_identities_sha256"]
        assert not protocol["graph_uses_validation_rows"] and not protocol["graph_uses_labels"]
        assert not protocol["competition_submission"] and not protocol["test_data_used"]
        paths = {
            method: directory / method / "validation_predictions.npz" for method in METHODS
        }
        if fold == "primary":
            paths["previous-best-ensemble"] = ROOT / "artifacts/local-t016/validation_predictions.npz"
        identities, labels, probabilities, fingerprints = load_candidates(
            paths, directory / "validation-identities.parquet",
        )
        scores = {
            name: float(roc_auc_score(labels, values)) for name, values in probabilities.items()
        }
        for method in METHODS:
            result = json.loads((directory / method / "results.json").read_text())
            assert result["selection_mode"] == "fixed_budget" and result["epochs_run"] == 8
            assert result["train_rows"] == counts[0] and result["validation_rows"] == counts[1]
            assert abs(result["validation_auc"] - scores[method]) < 1e-12
        comparisons = {}
        for reference, candidate in [
            ("tabm-control", "history-tabm"),
            ("tabm-control", "matrix-factorization"),
            ("tabm-control", "node-only-tabm"),
            ("tabm-control", "graphsage-tabm"),
            ("node-only-tabm", "graphsage-tabm"),
        ]:
            key = f"{candidate}-minus-{reference}"
            comparisons[key] = {
                "delta_auc": scores[candidate] - scores[reference],
                **paired_group_bootstrap(
                    labels, probabilities[reference], probabilities[candidate],
                    identities["request_id"].to_numpy(), repetitions=200,
                ),
            }
        coverage = {}
        with np.load(directory / "validation-entity-coverage.npz") as arrays:
            np.testing.assert_array_equal(arrays["row_ids"], identities["_row_id"].to_numpy())
            masks = {
                "both-seen": arrays["left_seen"] & arrays["right_seen"],
                "left-unseen": ~arrays["left_seen"],
                "right-unseen": ~arrays["right_seen"],
            }
            for name, mask in masks.items():
                coverage[name] = {
                    "rows": int(mask.sum()),
                    "auc": {
                        method: float(roc_auc_score(labels[mask], values[mask]))
                        if len(np.unique(labels[mask])) == 2 else None
                        for method, values in probabilities.items()
                    },
                }
        all_results[fold] = {
            "scores": scores, "paired_comparisons": comparisons,
            "entity_coverage": coverage, "prediction_sources": fingerprints,
            "training_rows": counts[0], "validation_rows": counts[1],
            "training_seconds": {
                method: json.loads((directory / method / "results.json").read_text())["training_seconds"]
                for method in METHODS
            },
        }
    report = {
        "status": "VERIFIED", "folds": all_results,
        "competition_submission": False,
        "interpretation": "Two complete chronological folds, fixed budgets; bootstrap conditional on selected recipes, not seed/time uncertainty.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "analysis-results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({fold: result["scores"] for fold, result in all_results.items()}, indent=2))


if __name__ == "__main__":
    main()
