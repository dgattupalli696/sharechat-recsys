from datetime import datetime, timedelta
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import polars as pl

from local_tree_experiment import CUTOFF, prepare_arrays, run
from train import features


class LocalTreeTests(unittest.TestCase):
    def test_fixed_budget_disk_worker_matches_in_memory_tree(self):
        from full_experiments import fit_stacking_base
        from tree_experiments import run_trees

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cutoff = datetime.fromisoformat(CUTOFF)
            count = 800
            rng = np.random.default_rng(42)
            signal = rng.normal(size=count)
            path = root / "train.parquet"
            pl.DataFrame({
                "request_id": [f"r{i // 4}" for i in range(count)],
                "feature1": [cutoff + timedelta(seconds=i - 640) for i in range(count)],
                "feature5": [f"category-{i % 3}" for i in range(count)],
                "feature33": signal,
                "feature26": rng.normal(size=(count, 32)).tolist(),
                "watched": (signal + rng.normal(size=count) > 0).astype(np.int8),
            }).write_parquet(path)
            base = root / "disk"
            report = fit_stacking_base(
                path, base, {"tree_kind": "lightgbm", "tree_iterations": 3},
                CUTOFF, None, "tree", device="cpu",
            )
            disk = base / "tree"
            self.assertEqual(report["results"][0]["selection_mode"], "fixed_budget")
            self.assertEqual(report["results"][0]["best_iteration"], 3)
            self.assertEqual(
                sum(line.startswith("Tree=") for line in (disk / "lightgbm/model.txt").read_text().splitlines()),
                3,
            )
            memory = root / "memory"
            memory.mkdir()
            run_trees(
                path, memory, CUTOFF, device="cpu", light_iterations=3,
                only=["lightgbm"], fixed_budget=True,
            )
            actual = np.load(disk / "lightgbm/validation_predictions.npz")
            expected = np.load(memory / "lightgbm/validation_predictions.npz")
            np.testing.assert_array_equal(actual["row_ids"], expected["row_ids"])
            np.testing.assert_array_equal(actual["y_true"], expected["y_true"])
            np.testing.assert_allclose(
                actual["probabilities"], expected["probabilities"], rtol=1e-12, atol=1e-12,
            )
            self.assertFalse(list(disk.glob("streaming-lightgbm-*")))

    def test_streaming_matches_full_features_and_preserves_reference_row_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cutoff = datetime.fromisoformat(CUTOFF)
            count = 160
            raw = pl.DataFrame({
                "request_id": [f"r{i // 4}" for i in range(count)],
                "feature1": [cutoff + timedelta(seconds=i - 128) for i in range(count)],
                "feature5": ["common" if i < 128 else "validation-only" for i in range(count)],
                "feature33": [float(i % 30) for i in range(count)],
                "feature26": [[float(i + j) for j in range(32)] for i in range(count)],
                "watched": [i % 2 for i in range(count)],
            }).reverse()
            path = root / "train.parquet"
            raw.write_parquet(path)
            expected = raw.with_row_index("_row_id").filter(pl.col("feature1") >= cutoff)
            reference = root / "reference.parquet"
            expected.select("_row_id", "request_id", "feature1").write_parquet(reference)
            cache, output = root / "cache", root / "prepared"
            cache.mkdir()
            output.mkdir()
            protocol = prepare_arrays(path, cache, output, reference)
            self.assertEqual(protocol["training_rows"], 128)
            self.assertEqual(protocol["validation_rows"], 32)
            matrix = np.load(cache / "valid.npy")
            columns = json.loads((cache / "metadata.json").read_text())["columns"]
            original = features(expected.drop("_row_id"), expand_embeddings=True)
            for name in ["feature33", "feature26_mean", "feature26_dim_7"]:
                np.testing.assert_array_equal(matrix[:, columns.index(name)], original[name].to_numpy())
            np.testing.assert_array_equal(matrix[:, columns.index("feature5")], -np.ones(32))
            report = run(path, root / "fit", iterations=2, threads=1, reference_identity=reference)
            predictions = np.load(root / "fit/lightgbm/validation_predictions.npz")
            np.testing.assert_array_equal(predictions["row_ids"], expected["_row_id"].to_numpy())
            self.assertTrue(np.isfinite(report["results"][0]["validation_auc"]))
            self.assertFalse(list((root / "fit").glob("streaming-lightgbm-*")))
            earlier_cache, earlier_output = root / "earlier-cache", root / "earlier-output"
            earlier_cache.mkdir()
            earlier_output.mkdir()
            earlier = prepare_arrays(
                path, earlier_cache, earlier_output,
                cutoff_string=(cutoff - timedelta(seconds=32)).isoformat(),
                validation_end=(cutoff + timedelta(seconds=16)).isoformat(),
            )
            self.assertEqual(earlier["training_rows"], 96)
            self.assertEqual(earlier["validation_rows"], 48)
            earlier_ids = pl.read_parquet(earlier_output / "validation-identities.parquet")
            self.assertTrue((earlier_ids["feature1"] < cutoff + timedelta(seconds=16)).all())
            self.assertTrue((earlier_ids["feature1"] >= cutoff - timedelta(seconds=32)).all())
