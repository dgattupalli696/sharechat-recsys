import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import polars as pl
from sklearn.metrics import log_loss, roc_auc_score


ROOT = Path.cwd()
if not (ROOT / "experiment_analysis.py").is_file():
    raise FileNotFoundError("Run from the repository root")
sys.path.insert(0, str(ROOT))
from experiment_analysis import load_candidates, paired_group_bootstrap


ARTIFACT = ROOT / "artifacts/kaggle-t018"
OUTPUT = ROOT / "artifacts/local-t018-confirmation-analysis"
RECIPES = ["reference", "tabm-k32-plr-cross-masked", "dcnv2-r32-plr-cross-masked"]
SEEDS = [42, 43, 44]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    expected = subprocess.check_output(
        ["git", "show", "88c26c4:experiment_analysis.py"], cwd=ROOT,
    )
    assert digest(ROOT / "experiment_analysis.py") == hashlib.sha256(expected).hexdigest()
    report = json.loads((ARTIFACT / "results.json").read_text())
    assert report["status"] == "COMPLETE" and len(report["results"]) == 18
    OUTPUT.mkdir(exist_ok=True)
    analysis = {
        "status": "COMPLETE", "competition_submission": False,
        "scope": "fixed three-seed means and preselected 75% k32 / 25% DCN vote; no weight search",
        "analysis_versions": {
            name: version(name) for name in ["numpy", "polars", "scikit-learn"]
        },
        "folds": {}, "seed42_reproduction": {},
        "source_sha256": {
            "analysis_script": digest(Path(__file__)),
            "experiment_analysis.py": digest(ROOT / "experiment_analysis.py"),
            "gpu_results": digest(ARTIFACT / "results.json"),
        },
        "training_seconds": sum(row["training_seconds"] for row in report["results"]),
        "preparation_seconds": sum(fold["preparation_seconds"] for fold in report["folds"]),
    }
    for recipe, batch, old_name in [
        ("reference", "kaggle-t008", "reference"),
        (RECIPES[1], "kaggle-t014", "ensemble-32"),
        (RECIPES[2], "kaggle-t014", "dcnv2-r32"),
    ]:
        previous = ROOT / "artifacts" / batch / old_name / "validation_predictions.npz"
        current = ARTIFACT / "primary" / f"{recipe}-seed-42/validation_predictions.npz"
        with np.load(previous) as old, np.load(current) as new:
            for key in ["y_true", "probabilities"]:
                np.testing.assert_array_equal(old[key], new[key])
            previous_ids = (
                old["row_ids"] if "row_ids" in old
                else pl.read_parquet(ROOT / "artifacts" / batch / "validation-identities.parquet")["_row_id"].to_numpy()
            )
            np.testing.assert_array_equal(previous_ids, new["row_ids"])
        analysis["seed42_reproduction"][recipe] = {
            "exact_predictions_and_ids": True,
            "previous_sha256": digest(previous), "new_sha256": digest(current),
        }
    for fold in ["primary", "earlier"]:
        identity_path = ARTIFACT / fold / "validation-identities.parquet"
        candidates = {
            f"{recipe}-seed-{seed}": ARTIFACT / fold / f"{recipe}-seed-{seed}/validation_predictions.npz"
            for recipe in RECIPES for seed in SEEDS
        }
        identity, labels, values, fingerprints = load_candidates(candidates, identity_path)
        protocol = next(row for row in report["folds"] if row["name"] == fold)
        assert len(labels) == protocol["validation_rows"]
        assert hashlib.sha256(labels.astype(np.uint8).tobytes()).hexdigest() == protocol["validation_labels_sha256"]
        votes = {
            recipe: {f"{recipe}-seed-{seed}": 1 / 3 for seed in SEEDS}
            for recipe in RECIPES
        }
        votes["frozen-75-25"] = {
            **{f"{RECIPES[1]}-seed-{seed}": 0.75 / 3 for seed in SEEDS},
            **{f"{RECIPES[2]}-seed-{seed}": 0.25 / 3 for seed in SEEDS},
        }
        probabilities = {}
        destination = OUTPUT / fold
        destination.mkdir(exist_ok=True)
        result = {
            "validation_rows": len(labels), "candidates": fingerprints,
            "identity_sha256": digest(identity_path), "seed_means": {},
        }
        for name, weights in votes.items():
            prediction = sum(weight * values[model].astype(np.float64) for model, weight in weights.items())
            probabilities[name] = prediction
            np.savez_compressed(
                destination / f"{name}.npz", probabilities=prediction,
                y_true=labels, row_ids=identity["_row_id"].to_numpy(),
            )
            result["seed_means"][name] = {
                "validation_auc": float(roc_auc_score(labels, prediction)),
                "validation_logloss": float(log_loss(labels, prediction, labels=[0, 1])),
                "weights": weights,
            }
        result["frozen_vote_per_seed"] = [
            {
                "seed": seed,
                "validation_auc": float(roc_auc_score(
                    labels, 0.75 * values[f"{RECIPES[1]}-seed-{seed}"].astype(np.float64)
                    + 0.25 * values[f"{RECIPES[2]}-seed-{seed}"].astype(np.float64),
                )),
            }
            for seed in SEEDS
        ]
        result["paired_request_bootstrap"] = {
            name: paired_group_bootstrap(
                labels, reference, candidate, identity["request_id"].to_numpy(),
            )
            for name, reference, candidate in [
                ("tabm_seed_mean_vs_reference_seed_mean", probabilities[RECIPES[0]], probabilities[RECIPES[1]]),
                ("frozen_vote_vs_tabm_seed_mean", probabilities[RECIPES[1]], probabilities["frozen-75-25"]),
            ]
        }
        if fold == "primary":
            previous_path = ROOT / "artifacts/local-t016/validation_predictions.npz"
            with np.load(previous_path) as previous:
                np.testing.assert_array_equal(labels, previous["y_true"])
                np.testing.assert_array_equal(identity["_row_id"].to_numpy(), previous["row_ids"])
                result["paired_request_bootstrap"]["frozen_vote_vs_t016"] = paired_group_bootstrap(
                    labels, previous["probabilities"], probabilities["frozen-75-25"],
                    identity["request_id"].to_numpy(),
                )
            result["previous_t016_sha256"] = digest(previous_path)
        analysis["folds"][fold] = result
    gpu_provenance = json.loads((Path(__file__).parent / "provenance.json").read_text())
    selected = next(
        row for row in report["results"]
        if row["fold"] == "primary" and row["recipe"] == RECIPES[1] and row["seed"] == 42
    )
    previous_result = json.loads((ROOT / "artifacts/local-t016/results.json").read_text())
    finalists = {
        "scope": "validation checkpoints; no competition-test inference or final all-row refit",
        "competition_submission": False,
        "submission_requires_explicit_approval": True,
        "selected_neural_family": {
            "recipe": {key: value for key, value in selected["spec"].items() if key != "name"},
            "seeds": SEEDS, "source_revision": "88c26c4",
            "source_sha256": gpu_provenance["source_sha256"],
            "summary": [row for row in report["summary"] if row["recipe"] == RECIPES[1]],
        },
        "best_observed_primary_holdout": {
            "trial": "T016", "validation_auc": previous_result["best_ensemble"]["validation_auc"],
            "flat_weights": {"tabm_k32_seed42": 0.45, "tabm_k8_seed42": 0.225,
                             "dcnv2_r32_seed42": 0.225, "lightgbm_seed42": 0.10},
            "prediction_sha256": digest(ROOT / "artifacts/local-t016/validation_predictions.npz"),
            "exact_four_component_mixture_confirmed_on_earlier_fold": False,
        },
        "two_fold_confirmed_frozen_vote": {
            "family_weights": {"tabm_k32": 0.75, "dcnv2_r32": 0.25},
            "seeds_per_family": SEEDS,
            "member_weights": analysis["folds"]["primary"]["seed_means"]["frozen-75-25"]["weights"],
            "folds": {
                fold: {
                    "validation_auc": analysis["folds"][fold]["seed_means"]["frozen-75-25"]["validation_auc"],
                    "prediction_path": str((OUTPUT / fold / "frozen-75-25.npz").relative_to(ROOT)),
                    "prediction_sha256": digest(OUTPUT / fold / "frozen-75-25.npz"),
                    "checkpoint_sha256": {
                        name: gpu_provenance["artifact_sha256"][f"{fold}/{name}/best_model.pt"]
                        for name in analysis["folds"][fold]["seed_means"]["frozen-75-25"]["weights"]
                    },
                }
                for fold in ["primary", "earlier"]
            },
        },
        "interpretation": "Scores and intervals are conditional, not untouched future-test confirmation.",
    }
    (OUTPUT / "finalists.json").write_text(json.dumps(finalists, indent=2) + "\n")
    (OUTPUT / "results.json").write_text(json.dumps(analysis, indent=2) + "\n")
    print(json.dumps({
        fold: {
            "seed_means": result["seed_means"],
            "frozen_vote_per_seed": result["frozen_vote_per_seed"],
            "paired_request_bootstrap": result["paired_request_bootstrap"],
        }
        for fold, result in analysis["folds"].items()
    }, indent=2))


if __name__ == "__main__":
    main()
