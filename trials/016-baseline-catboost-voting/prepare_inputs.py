import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts/local-t016"


def main():
    manifest = json.loads((ROOT / "artifacts/local-t015/candidates.json").read_text())
    baseline = json.loads((ROOT / "artifacts/b001-validation-replay/results.json").read_text())
    assert baseline["trees"] == 21 and baseline["input_features"] == 679
    assert baseline["training_rows"] == 1230110 and baseline["validation_rows"] == 307528
    assert not baseline["retraining"]
    assert abs(baseline["validation_auc"] - 0.8160260803) < 1e-8
    manifest["candidates"]["catboost-baseline"] = "artifacts/b001-validation-replay/validation_predictions.npz"
    manifest["families"]["catboost-baseline"] = "CatBoost"
    manifest["candidates"]["current-ensemble"] = "artifacts/local-t015/validation_predictions.npz"
    manifest["families"]["current-ensemble"] = "Current ensemble"
    manifest["core_names"] = list(manifest["candidates"])
    manifest["pair_weights"] = [
        0.01, 0.025, 0.05, 0.1, 0.125, 0.25, 0.5,
        0.75, 0.875, 0.9, 0.95, 0.975, 0.99,
    ]
    manifest["four_way_groups"].extend([
        [tabm, "dcnv2-moe4", finalmlp, "catboost-baseline"]
        for tabm in ["tabm-k8", "tabm-k16", "tabm-k32"]
        for finalmlp in ["finalmlp-gated", "finalmlp-ungated", "finalmlp-alternate-lr"]
    ])
    manifest["scope"] = "Add original pre-refit CatBoost baseline and frozen current winner to complete primary-holdout soft voting"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "candidates.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(manifest['candidates'])} full-holdout voting candidates")


if __name__ == "__main__":
    main()
