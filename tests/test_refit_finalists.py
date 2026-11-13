from datetime import datetime, timedelta
from importlib.metadata import version
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import shutil
import subprocess

import numpy as np
import polars as pl
import torch

from custom_network import prepare_custom_features
from evaluate_refit import evaluate
from refit_finalists import digest, prepare, run
from train import features


class FullRefitTests(unittest.TestCase):
    def fixture(self, root):
        rows = 160
        frame = pl.DataFrame({
            "request_id": [f"r-{i}" for i in range(rows)],
            "feature1": [datetime(2026, 10, 25) + timedelta(seconds=i) for i in range(rows)],
            "feature5": ["last-only" if i >= 150 else f"group-{i % 3}" for i in range(rows)],
            "feature33": [None if i % 7 == 0 else float(i) for i in range(rows)],
            "feature26": [[float((i + j) % 4) for j in range(32)] for i in range(rows)],
            "watched": [i % 2 for i in range(rows)],
        })
        frame.write_parquet(root / "train.parquet")
        test = frame.head(20).drop("request_id", "watched").with_columns(
            pl.Series("id", np.arange(1000, 1020)),
            pl.lit("test-only").alias("feature5"),
        )
        test.write_parquet(root / "test.parquet")
        pl.DataFrame({"id": np.arange(1000, 1020), "watched": [0.] * 20}).write_csv(root / "sample_submission.csv")
        return frame, test

    def test_full_preprocessing_matches_eager_recipe_and_includes_latest_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame, test = self.fixture(root)
            cache, output = root / "cache", root / "output"
            cache.mkdir()
            output.mkdir()
            x, c, labels, order, state, config = prepare(root / "train.parquet", cache, output)
            expected_x, expected_c, _, _, expected = prepare_custom_features(
                features(frame, True), features(test, True), 10,
            )
            torch.testing.assert_close(torch.from_numpy(x), expected_x, rtol=1e-5, atol=1e-5)
            np.testing.assert_array_equal(c, expected_c.numpy())
            np.testing.assert_array_equal(labels, frame["watched"].to_numpy())
            np.testing.assert_array_equal(order, np.arange(160))
            self.assertIn("last-only", state["encoder"].categories_[0])
            self.assertNotIn("test-only", state["encoder"].categories_[0])
            np.testing.assert_allclose(state["imputer"].statistics_, expected["imputer"].statistics_)
            self.assertEqual(config["encoder_config"]["n_numeric"], x.shape[1])

    def test_full_refit_exports_ordered_fixed_weight_csvs_without_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            output = root / "output"
            selected = {
                "full_training_rows": 160, "lightgbm_trees": 2,
                "ensemble_weights": {"tabm32": .45, "tabm8": .225, "dcn32": .225, "lightgbm": .1},
                "neural": {
                    "tabm32": {"members": 2, "architecture": "tabm", "fractional_epochs": 1.},
                    "tabm8": {"members": 2, "architecture": "tabm", "fractional_epochs": 1.},
                    "dcn32": {"members": 2, "architecture": "dcnv2", "fractional_epochs": 1.},
                },
            }
            report = run(root, output, selected, device="cpu")
            self.assertIsNone(report["validation_auc"])
            self.assertFalse(report["validation_used_for_refit_selection"])
            self.assertFalse(report["competition_submission"])
            for result in report["neural"].values():
                self.assertEqual(result["training_rows"], 160)
                self.assertEqual(result["optimizer_steps"], 1)
            self.assertFalse((output / "lightgbm/validation_predictions.npz").exists())
            worker = json.loads((output / "lightgbm/worker-results.json").read_text())
            self.assertTrue(worker["full_training"])
            self.assertEqual(worker["training_rows"], 160)
            with np.load(output / "test-component-predictions.npz") as arrays:
                expected = sum(report["ensemble_weights"][name] * arrays[name].astype(float) for name in report["ensemble_weights"])
            actual = pl.read_csv(output / "submission-full-ensemble.csv")
            np.testing.assert_array_equal(actual["id"].to_numpy(), np.arange(1000, 1020))
            np.testing.assert_allclose(actual["watched"].to_numpy(), expected, rtol=0, atol=1e-15)
            self.assertTrue((output / "submission-full-tabm.csv").is_file())
            bundle = root / "bundle"
            bundle.mkdir()
            for name in selected["neural"]:
                shutil.copyfile(output / name / "model.pt", bundle / f"{name}.pt")
            for filename in ["preprocessing.joblib", "category-vocabularies.json"]:
                shutil.copyfile(output / filename, bundle / filename)
            shutil.copyfile(output / "lightgbm/model.txt", bundle / "lightgbm.txt")
            evaluation_ids = np.arange(80, 160, dtype=np.uint32)[::-1]
            np.savez_compressed(
                bundle / "evaluation-reference.npz",
                row_ids=evaluation_ids, y_true=(evaluation_ids % 2).astype(np.int8),
            )
            manifest = {
                "files_sha256": {path.name: digest(path) for path in bundle.iterdir()},
                "versions": {name: version(name) for name in ["torch", "numpy", "scikit-learn"]},
                "ensemble_weights": selected["ensemble_weights"],
                "cutoff": (datetime(2026, 10, 25) + timedelta(seconds=80)).isoformat(),
                "checkpoint_training_rows": 160,
            }
            (bundle / "manifest.json").write_text(json.dumps(manifest))
            sanity = root / "sanity"
            with patch("torch.optim.AdamW", side_effect=AssertionError("Evaluation must not train")), patch(
                "evaluate_refit.subprocess.run", wraps=subprocess.run,
            ) as calls:
                sanity_report = evaluate(bundle, root / "train.parquet", sanity, device="cpu")
            self.assertFalse(sanity_report["additional_training"])
            self.assertFalse(sanity_report["unbiased_validation"])
            self.assertFalse(sanity_report["used_for_model_selection"])
            self.assertEqual(sanity_report["training_overlap_rows"], 80)
            self.assertEqual(set(sanity_report["scores"]), {*selected["ensemble_weights"], "ensemble"})
            self.assertEqual(calls.call_count, 1)
            self.assertIn("--predict-array", calls.call_args.args[0])
            self.assertNotIn("--full-training", calls.call_args.args[0])
            with np.load(sanity / "former-holdout-predictions.npz") as arrays:
                np.testing.assert_array_equal(arrays["row_ids"], evaluation_ids)


if __name__ == "__main__":
    unittest.main()
