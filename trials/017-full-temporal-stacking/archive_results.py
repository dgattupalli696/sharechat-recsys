import ast
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys

import nbformat
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
import torch


ROOT = Path.cwd()
if not (ROOT / "full_experiments.py").is_file():
    raise FileNotFoundError("Run this archival script from the repository root")
sys.path.insert(0, str(ROOT))
from experiment_analysis import paired_group_bootstrap
from full_experiments import finish_stacking


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_snapshot(trial, artifact, revision, expected_version):
    provenance = json.loads((trial / "provenance.json").read_text())
    source = artifact.with_name(artifact.name + "-source")
    assert digest(trial / "notebook.ipynb") == provenance["notebook_sha256"]
    assert digest(trial / "kernel-metadata.json") == provenance["configuration_sha256"]
    for name, expected in provenance["source_sha256"].items():
        payload = subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=ROOT)
        assert hashlib.sha256(payload).hexdigest() == expected
    prepared = nbformat.read(trial / "notebook.ipynb", as_version=4)
    exported = nbformat.read(next(source.glob("*.ipynb")), as_version=4)
    assert [cell.source for cell in prepared.cells] == [cell.source for cell in exported.cells]
    requested = json.loads((trial / "kernel-metadata.json").read_text())
    metadata = json.loads((source / "kernel-metadata.json").read_text())
    for key in ["id", "docker_image", "enable_gpu", "enable_internet", "is_private"]:
        assert requested[key] == metadata[key]
    assert int(artifact.with_name(artifact.name + "-version.txt").read_text()) == expected_version
    return provenance, metadata


def main():
    trial = ROOT / "trials/017-full-temporal-stacking"
    original = ROOT / "artifacts/kaggle-t017"
    recovery = ROOT / "artifacts/kaggle-t017-recovery"
    local = ROOT / "artifacts/local-t017-tree"
    local_trial = trial / "recovery/local-tree"
    output = ROOT / "artifacts/local-t017-stacking"
    first, first_metadata = verify_snapshot(trial, original, "b7ee803", 7)
    second, second_metadata = verify_snapshot(trial / "recovery", recovery, "d96f4a1", 8)
    assert first_metadata["id_no"] == second_metadata["id_no"]
    for name, expected in first["source_sha256"].items():
        if name != "full_experiments.py":
            assert second["source_sha256"][name] == expected
    definitions = []
    for revision in ["b7ee803", "d96f4a1"]:
        source = subprocess.check_output(["git", "show", f"{revision}:full_experiments.py"], cwd=ROOT)
        definitions.append({
            node.name: ast.dump(node)
            for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)
        })
    for name in ["prepare_data", "category_view", "run_tabm"]:
        assert definitions[0][name] == definitions[1][name]
    current = {
        node.name: ast.dump(node)
        for node in ast.parse((ROOT / "full_experiments.py").read_text()).body
        if isinstance(node, ast.FunctionDef)
    }
    assert current["finish_stacking"] == definitions[1]["finish_stacking"]
    third = json.loads((local_trial / "provenance.json").read_text())
    assert third["source_sha256"]["train.py"] == first["source_sha256"]["train.py"]
    for name, expected in third["source_sha256"].items():
        assert digest(local_trial / "frozen" / name) == expected
        payload = subprocess.check_output(["git", "show", f"0e43e5b:{name}"], cwd=ROOT)
        assert hashlib.sha256(payload).hexdigest() == expected
    old = json.loads((original / "results.json").read_text())
    new = json.loads((local / "results.json").read_text())
    assert len(old["folds"]) == 3 and "status" not in old
    assert new["status"] == "COMPLETE" and new["stage"] == "local-trees"
    assert new["results"][0]["selection_mode"] == "fixed_budget"
    outer_protocol = json.loads((recovery / "outer/protocol.json").read_text())
    for key in ["training_rows", "validation_rows", "cutoff", "validation_labels_sha256"]:
        assert outer_protocol[key] == new["protocol"][key]
    outer_fold = {
        "cutoff": outer_protocol["cutoff"], "end": None, "protocol": outer_protocol,
        "neural": json.loads((recovery / "outer/neural-fixed-budget/results.json").read_text()),
        "tree": new["results"][0],
    }
    assert digest(recovery / "outer/validation-identities.parquet") == third["reference_identity_sha256"]
    output.mkdir(exist_ok=True)
    for index in range(1, 4):
        shutil.copytree(original / f"oof-{index}", output / f"oof-{index}", dirs_exist_ok=True)
    shutil.copytree(recovery / "outer", output / "outer", dirs_exist_ok=True)
    shutil.copytree(local / "lightgbm", output / "outer/tree/lightgbm", dirs_exist_ok=True)
    shutil.copyfile(local / "results.json", output / "outer/tree/results.json")
    report = {
        "stage": "stacking", "folds": old["folds"] + [outer_fold],
        "fixed_neural_epochs": 8, "competition_submission": False,
        "source_batches": [
            {"version": 7, "revision": "b7ee803", "status": "ERROR", "completed_base_fits": 6},
            {"version": 8, "revision": "d96f4a1", "status": "ERROR", "completed_base_fits": 1},
            {"location": "local", "revision": "0e43e5b", "status": "COMPLETE", "completed_base_fits": 1},
        ],
        "meta_environment": {
            "location": "local", "versions": {
                name: version(name) for name in ["numpy", "scikit-learn", "joblib"]
            },
        },
    }
    raw = pl.read_parquet(ROOT / "data/train.parquet", columns=["feature1", "watched"])
    for index, fold in enumerate(report["folds"]):
        destination = output / ("outer" if fold["end"] is None else f"oof-{index + 1}")
        cutoff = datetime.fromisoformat(fold["cutoff"])
        training = (raw["feature1"] < cutoff).to_numpy()
        validation = ~training
        if fold["end"]:
            validation &= (raw["feature1"] < datetime.fromisoformat(fold["end"])).to_numpy()
        protocol = fold["protocol"]
        assert int(training.sum()) == protocol["training_rows"]
        assert int(validation.sum()) == protocol["validation_rows"]
        for folder, key in [("neural-fixed-budget", "neural"), ("tree/lightgbm", "tree")]:
            path = destination / folder
            result = fold[key]
            with np.load(path / "validation_predictions.npz") as predictions:
                np.testing.assert_array_equal(np.sort(predictions["row_ids"]), np.flatnonzero(validation))
                np.testing.assert_array_equal(
                    predictions["y_true"], raw["watched"].to_numpy()[predictions["row_ids"]],
                )
                probabilities = predictions["probabilities"]
                assert np.isfinite(probabilities).all()
                assert ((probabilities >= 0) & (probabilities <= 1)).all()
                assert abs(roc_auc_score(predictions["y_true"], probabilities) - result["validation_auc"]) < 1e-12
            if key == "neural":
                saved = torch.load(path / "best_model.pt", map_location="cpu", weights_only=True)
                assert saved["model_config"] == result["model_config"]
                assert all(torch.isfinite(value).all() for value in saved["state_dict"].values())
                assert result["selection_mode"] == "fixed_budget" and len(result["checkpoint_history"]) == 1
                assert result["optimizer_steps"] == result["best_global_step"] == math.ceil(training.sum() / 1024) * 8
            else:
                assert result["best_iteration"] == 140
                assert sum(line.startswith("Tree=") for line in (path / "model.txt").read_text().splitlines()) == 140
    report = finish_stacking(output, report)
    assert report["oof_rows"] == 943867 and report["validation_rows"] == 307528
    identity = pl.read_parquet(output / "outer/validation-identities.parquet")
    outer = np.load(output / "validation_predictions.npz")
    neural = np.load(output / "outer/neural-fixed-budget/validation_predictions.npz")
    tree = np.load(output / "outer/tree/lightgbm/validation_predictions.npz")
    previous_tree = np.load(ROOT / "artifacts/local-t011/lightgbm/validation_predictions.npz")
    for key in ["y_true", "row_ids", "probabilities"]:
        np.testing.assert_array_equal(tree[key], previous_tree[key])
    report["tree_endpoint_reproduces_t011"] = {
        "predictions_exactly_equal": True,
        "previous_prediction_sha256": digest(
            ROOT / "artifacts/local-t011/lightgbm/validation_predictions.npz"
        ),
    }
    np.testing.assert_array_equal(identity["_row_id"].to_numpy(), outer["row_ids"])
    report["paired_request_bootstrap"] = {
        label: paired_group_bootstrap(
            outer["y_true"], reference, outer["probabilities"], identity["request_id"].to_numpy(),
        )
        for label, reference in [
            ("versus_matched_neural", neural["probabilities"]),
            ("versus_matched_equal_blend", (neural["probabilities"] + tree["probabilities"]) / 2),
        ]
    }
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    shutil.copyfile(output / "results.json", trial / "results.json")
    for filename in ["results.json", "protocol.json", "run.log"]:
        shutil.copyfile(local / filename, local_trial / filename)
    second.update({
        "outer_neural_verified": True,
        "artifact_sha256": {
            str(path.relative_to(recovery)): digest(path)
            for path in sorted(recovery.rglob("*")) if path.is_file()
        },
    })
    third.update({
        "source_revision": "0e43e5b", "status": "COMPLETE",
        "artifact_sha256": {
            str(path.relative_to(local)): digest(path)
            for path in sorted(local.rglob("*")) if path.is_file()
        },
    })
    first.update({
        "status": "COMPLETE_WITH_RECOVERY", "source_batches": report["source_batches"],
        "local_assembly": str(output.relative_to(ROOT)), "meta_environment": report["meta_environment"],
        "assembly_source_sha256": digest(Path(__file__)),
        "assembled_artifact_sha256": {
            str(path.relative_to(output)): digest(path)
            for path in sorted(output.rglob("*")) if path.is_file()
        },
    })
    (trial / "recovery/provenance.json").write_text(json.dumps(second, indent=2) + "\n")
    (local_trial / "provenance.json").write_text(json.dumps(third, indent=2) + "\n")
    (trial / "provenance.json").write_text(json.dumps(first, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "folds"}, indent=2))


if __name__ == "__main__":
    main()
