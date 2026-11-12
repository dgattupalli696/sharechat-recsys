from pathlib import Path
import tempfile
import unittest

import numpy as np
import polars as pl

from ensemble_experiment import run_ensembles


class EnsembleTests(unittest.TestCase):
    def test_complementary_averaging_improves_auc_without_duplicate_weight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels, ids = np.array([0, 0, 1, 1]), np.arange(4)
            identity = root / "identity.parquet"
            pl.DataFrame({"_row_id": ids, "request_id": ["a", "b", "c", "d"]}).write_parquet(identity)
            candidates = {}
            for name, probabilities in [
                ("first", [0.1, 0.8, 0.7, 0.9]),
                ("duplicate", [0.1, 0.8, 0.7, 0.9]),
                ("second", [0.8, 0.1, 0.9, 0.7]),
                ("third", [0.2, 0.9, 0.8, 0.9]),
                ("fourth", [0.9, 0.2, 0.9, 0.8]),
            ]:
                path = root / f"{name}.npz"
                np.savez(path, y_true=labels, probabilities=probabilities, row_ids=ids)
                candidates[name] = path
            report = run_ensembles(
                candidates, identity, root / "results", core_names=list(candidates),
                families={
                    "first": "neural", "duplicate": "neural", "second": "tree",
                    "third": "cross", "fourth": "fusion",
                },
                four_way_groups=[["first", "second", "third", "fourth"]],
                pair_weights=[0.1, 0.5, 0.9],
                repetitions=10,
            )
            self.assertEqual(report["unique_candidates"], 4)
            self.assertEqual(report["duplicate_aliases"], {"duplicate": "first"})
            self.assertEqual(report["best_equal_weight"]["validation_auc"], 1)
            self.assertEqual(report["standalone_scores"]["first"], 0.75)
            mixtures = [
                row for row in report["configurations"] if row["kind"] == "four-family-grid"
            ]
            self.assertEqual(max(row["validation_auc"] for row in mixtures), 1)
            self.assertTrue(all(
                abs(sum(row["weights"]) - 1) < 1e-12 and min(row["weights"]) > 0
                for row in mixtures
            ))
            self.assertTrue(any(
                row["kind"] == "pair-grid" and row["weights"] == [0.1, 0.9]
                for row in report["configurations"]
            ))
            predictions = np.load(root / "results/equal_weight_predictions.npz")
            np.testing.assert_array_equal(predictions["row_ids"], ids)
