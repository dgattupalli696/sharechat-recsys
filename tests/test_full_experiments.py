from datetime import datetime, timedelta
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import polars as pl
import torch

from full_experiments import PRIMARY_CUTOFF, category_view, feature_specs, finish_stacking, prepare_data, run_confirmation, run_stacking, run_tabm
from tree_experiments import run_trees
from tabm_experiment import TabMClassifier, fit_model, fit_model_steps


class FullExperimentTests(unittest.TestCase):
    def test_fine_checks_preserve_optimizer_trajectory_and_training_mode(self):
        torch.set_num_threads(1)
        numeric = torch.randn(64, 3, generator=torch.Generator().manual_seed(7))
        categories = torch.ones(64, 1, dtype=torch.int64)
        target = torch.tensor([0.0, 1.0] * 32)
        trajectories = []
        with tempfile.TemporaryDirectory() as directory:
            for fine in [False, True]:
                torch.manual_seed(42)
                config = {
                    "n_numeric": 3, "cardinalities": [3], "members": 2,
                    "width": 8, "category_dropout": 0.3,
                }
                model = TabMClassifier(**config)
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
                original_step = optimizer.step
                states = []

                def step(*args, **kwargs):
                    self.assertTrue(model.training)
                    result = original_step(*args, **kwargs)
                    states.append({key: value.clone() for key, value in model.state_dict().items()})
                    return result

                output = Path(directory) / str(fine)
                output.mkdir()
                with patch.object(optimizer, "step", step):
                    function = fit_model_steps if fine else fit_model
                    extra = {"validation_interval": 2} if fine else {}
                    result = function(
                        model, config, optimizer, numeric, categories, target,
                        numeric[:16], categories[:16], target[:16].numpy(),
                        device="cpu", output=output, epochs=1, patience=5,
                        batch_size=8, **extra,
                    )
                trajectories.append(states)
                if fine:
                    self.assertEqual(len(result["checkpoint_history"]), 4)
                    self.assertIn(result["best_global_step"], [2, 4, 6, 8])
                    chosen = states[result["best_global_step"] - 1]
                    for key, expected in chosen.items():
                        torch.testing.assert_close(model.state_dict()[key], expected)
            for original, fine in zip(*trajectories):
                for key in original:
                    torch.testing.assert_close(original[key], fine[key], rtol=0, atol=0)

    def test_full_partition_identity_and_category_remapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cutoff = datetime.fromisoformat(PRIMARY_CUTOFF)
            rows = 160
            pl.DataFrame({
                "request_id": [f"request-{i // 4}" for i in range(rows)],
                "feature1": [cutoff + timedelta(seconds=i - 128) for i in range(rows)],
                "feature5": ["common" if i % 3 else f"id-{i}" for i in range(rows)],
                "feature33": [float(i % 30) for i in range(rows)],
                "feature26": [[float((i + j) % 4) for j in range(32)] for i in range(rows)],
                "watched": [i % 2 for i in range(rows)],
            }).write_parquet(root / "train.parquet")
            output = root / "output"
            output.mkdir()
            data = prepare_data(root / "train.parquet", output)
            self.assertEqual(data["protocol"]["training_rows"], 128)
            self.assertEqual(data["protocol"]["validation_rows"], 32)
            identities = pl.read_parquet(output / "validation-identities.parquet")
            self.assertEqual(identities["_row_id"].to_list(), list(range(128, 160)))
            train, valid, cardinalities = category_view(data, minimum=1000)
            self.assertEqual(cardinalities, [1])
            self.assertTrue((train == 0).all().item())
            self.assertTrue((valid == 0).all().item())
            for name, changes in [
                ("reference", {}), ("adaptive", {"adaptive": True}),
                ("without-categories", {"drop_high": True}),
            ]:
                report = run_tabm(
                    data, {"name": name, "width": 8, "members": 2, **changes},
                    output, device="cpu", epochs=1, interval=1,
                )
                self.assertEqual(report["optimizer_steps"], 1)
                self.assertTrue(np.isfinite(report["validation_auc"]))
                self.assertFalse((output / name / "submission.csv").exists())
            for spec in feature_specs()[1:]:
                report = run_tabm(
                    data, {**spec, "width": 8, "members": 2},
                    output, device="cpu", epochs=1, interval=1,
                )
                self.assertEqual(report["optimizer_steps"], 1)
                self.assertTrue(np.isfinite(report["validation_auc"]))
            self.assertFalse(json.loads((output / "protocol.json").read_text())["competition_submission"])
            tree_output = root / "trees"
            tree_output.mkdir()
            report = run_trees(
                root / "train.parquet", tree_output, PRIMARY_CUTOFF,
                device="cpu", iterations=2, light_iterations=2,
            )
            self.assertEqual(len(report["results"]), 6)
            self.assertTrue(all(np.isfinite(result["validation_auc"]) for result in report["results"]))
            for result in report["results"]:
                arrays = np.load(tree_output / result["name"] / "validation_predictions.npz")
                np.testing.assert_array_equal(arrays["y_true"], data["y_valid"])

    def test_temporal_stacking_uses_disjoint_oof_blocks_and_fixed_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "chronological.parquet"
            n = 400
            start = datetime(2026, 10, 25, 18)
            vectors = np.random.default_rng(42).normal(size=(n, 32)).astype(np.float32)
            pl.DataFrame({
                "request_id": [f"r-{i // 4}" for i in range(n)],
                "feature1": [start + timedelta(minutes=i * 11) for i in range(n)],
                "feature5": ["common"] * n,
                "feature33": [float(i % 30) for i in range(n)],
                "feature26": vectors.tolist(),
                "watched": [i % 2 for i in range(n)],
            }).write_parquet(path)
            output = root / "stack"
            output.mkdir()
            recipe = {"name": "reference", "members": 2, "width": 8}
            report = run_stacking(
                path, output,
                {"neural_recipe": recipe, "fixed_neural_epochs": 1,
                 "tree_kind": "catboost-expanded", "tree_iterations": 2},
                device="cpu",
            )
            self.assertEqual(report["status"], "COMPLETE")
            self.assertEqual(len(report["folds"]), 4)
            for fold in report["folds"]:
                self.assertEqual(fold["neural"]["selection_mode"], "fixed_budget")
                self.assertEqual(len(fold["neural"]["checkpoint_history"]), 1)
            oof = np.load(output / "oof_predictions.npz")
            outer = np.load(output / "validation_predictions.npz")
            self.assertEqual(len(oof["row_ids"]), len(np.unique(oof["row_ids"])))
            self.assertFalse(np.intersect1d(oof["row_ids"], outer["row_ids"]).size)
            recovery = root / "outer-only"
            recovery.mkdir()
            recovered = run_stacking(
                path, recovery,
                {"neural_recipe": recipe, "fixed_neural_epochs": 1,
                 "tree_kind": "catboost-expanded", "tree_iterations": 2},
                device="cpu", outer_only=True,
            )
            self.assertEqual(recovered["stage"], "stacking-base-fits")
            self.assertEqual(len(recovered["folds"]), 1)
            self.assertEqual(recovered["folds"][0]["cutoff"], PRIMARY_CUTOFF)
            repeated = np.load(
                recovery / "outer/neural-fixed-budget/validation_predictions.npz"
            )
            original = np.load(output / "outer/neural-fixed-budget/validation_predictions.npz")
            np.testing.assert_array_equal(repeated["probabilities"], original["probabilities"])
            self.assertFalse((recovery / "meta-model.joblib").exists())
            with self.assertRaisesRegex(ValueError, "all three forward OOF"):
                finish_stacking(recovery, recovered)
            confirmation = root / "confirmation"
            confirmation.mkdir()
            checked = run_confirmation(
                path, confirmation, [recipe],
                device="cpu", epochs=1, interval=300,
            )
            self.assertEqual(len(checked["results"]), 6)
            self.assertEqual(len(checked["summary"]), 2)
            for fold in ["primary", "earlier"]:
                partial = json.loads((confirmation / f"{fold}-results.json").read_text())
                self.assertEqual(partial["status"], "COMPLETE")
                self.assertEqual([entry["name"] for entry in partial["folds"]], [fold])

    def test_stacking_surfaces_isolated_fit_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch(
                "full_experiments.subprocess.run",
                side_effect=subprocess.CalledProcessError(1, ["failed-worker"]),
            ) as worker:
                with self.assertRaises(subprocess.CalledProcessError):
                    run_stacking(
                        output / "train.parquet", output,
                        {"neural_recipe": {"name": "reference"},
                         "fixed_neural_epochs": 8, "tree_kind": "lightgbm",
                         "tree_iterations": 140},
                        device="cpu",
                    )
            worker.assert_called_once()
            self.assertTrue(worker.call_args.kwargs["check"])
            self.assertFalse((output / "results.json").exists())


if __name__ == "__main__":
    unittest.main()
