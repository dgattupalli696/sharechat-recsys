from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "artifacts/local-t013"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    reference = json.loads((ROOT / "trials/008-full-gpu-tabm-controls/provenance.json").read_text())
    identity_path = ROOT / "artifacts/kaggle-t008/validation-identities.parquet"
    assert digest(identity_path) == reference["artifact_sha256"]["validation-identities.parquet"]
    row_ids = pl.read_parquet(identity_path)["_row_id"].to_numpy()
    targets = np.load(
        ROOT / "artifacts/kaggle-t008/epoch-checkpoint-control/validation_predictions.npz"
    )["y_true"]
    image = json.loads(
        (ROOT / "trials/008-full-gpu-tabm-controls/executed-kernel-metadata.json").read_text()
    )["docker_image"]
    candidates, families, certificates = {}, {}, {}
    for batch in ["t008", "t009"]:
        directory = ROOT / "artifacts" / f"kaggle-{batch}"
        report = json.loads((directory / "results.json").read_text())
        assert report["status"] == "COMPLETE"
        assert report["protocol"]["training_rows"] == 1230110
        assert report["protocol"]["validation_rows"] == 307528
        for result in report["results"]:
            name = f"{batch}-{result['name']}"
            candidates[name] = str((directory / result["name"] / "validation_predictions.npz").relative_to(ROOT))
            families[name] = "TabM"
    legacy = [
        ("t006-attention", "006-full-gpu-vector-attention", "kaggle-t006", "attention", "Attention"),
        ("t007-standalone", "007-full-gpu-tabm-variants", "kaggle-t007", "standalone", "TabM"),
        ("t007-hybrid", "007-full-gpu-tabm-variants", "kaggle-t007", "hybrid", "Attention"),
    ]
    for name, trial_name, artifact_name, variant, family in legacy:
        trial = ROOT / "trials" / trial_name
        provenance = json.loads((trial / "provenance.json").read_text())
        report = json.loads((trial / "results.json").read_text())
        assert provenance["kaggle_status"] == "COMPLETE"
        assert provenance["docker_image"] == image
        assert provenance["source_sha256"]["train.py"] == reference["source_sha256"]["train.py"]
        assert provenance["artifact_sha256"]["preprocessing.joblib"] == reference["artifact_sha256"]["preprocessing.joblib"]
        assert report["sample_rows"] == 1537638
        assert report["training_rows"] == 1230110 and report["validation_rows"] == 307528
        assert datetime.fromisoformat(report["cutoff"]) == datetime.fromisoformat("2026-10-28T08:06:52.397000")
        path = ROOT / "artifacts" / artifact_name / variant / "validation_predictions.npz"
        expected = provenance["artifact_sha256"][f"{variant}/validation_predictions.npz"]
        assert digest(path) == expected
        arrays = np.load(path)
        np.testing.assert_array_equal(arrays["y_true"], targets)
        if name == "t007-standalone":
            np.testing.assert_array_equal(
                arrays["probabilities"],
                np.load(ROOT / "artifacts/kaggle-t008/epoch-checkpoint-control/validation_predictions.npz")["probabilities"],
            )
        destination = OUTPUT / "aligned-legacy" / name
        destination.mkdir(parents=True, exist_ok=True)
        aligned = destination / "validation_predictions.npz"
        np.savez_compressed(
            aligned, y_true=targets, probabilities=arrays["probabilities"], row_ids=row_ids,
        )
        candidates[name] = str(aligned.relative_to(ROOT))
        families[name] = family
        certificates[name] = {
            "original_path": str(path.relative_to(ROOT)), "original_sha256": expected,
            "aligned_sha256": digest(aligned),
            "canonical_identity_sha256": digest(identity_path),
            "common_ordered_loader_sha256": reference["source_sha256"]["train.py"],
            "common_preprocessing_sha256": reference["artifact_sha256"]["preprocessing.joblib"],
            "method": "Derived IDs: same archived full-data ordered loader, cutoff, image and byte-identical fitted preprocessing; labels checked. Legacy IDs were not stored originally.",
        }
    candidates["t011-lightgbm"] = "artifacts/local-t011/lightgbm/validation_predictions.npz"
    families["t011-lightgbm"] = "LightGBM"
    manifest = {
        "candidates": candidates, "families": families,
        "core_names": [
            "t009-vector-crosses", "t008-without-high-cardinality",
            "t008-cosine-two-epochs", "t009-plr-d8", "t009-vector-projection-ln",
            "t006-attention", "t007-hybrid", "t011-lightgbm",
        ],
        "legacy_alignment_certificates": certificates,
        "scope": "all currently archived full-data models with available validation predictions; never subset or earlier-fold predictions",
    }
    (OUTPUT / "candidates.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {len(candidates)} full-holdout candidates")


if __name__ == "__main__":
    main()
