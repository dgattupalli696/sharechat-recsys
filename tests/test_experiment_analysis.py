from pathlib import Path
import tempfile
import unittest

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from experiment_analysis import analyze, auc_order, paired_group_bootstrap, weighted_auc


class AnalysisTests(unittest.TestCase):
    def test_weighted_auc_handles_ties_like_the_reference_metric(self):
        labels = np.array([0, 1, 0, 1, 0, 1])
        values = np.array([0.1, 0.5, 0.5, 0.5, 0.9, 0.9])
        weights = np.array([1.0, 2, 0, 3, 1, 4])
        self.assertAlmostEqual(
            weighted_auc(auc_order(labels, values), weights),
            roc_auc_score(labels, values, sample_weight=weights),
        )

    def test_blending_preserves_row_alignment_and_includes_single_model_endpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = np.array([0, 1] * 8)
            ids = np.arange(16)
            identity = root / "identity.parquet"
            pl.DataFrame({
                "_row_id": ids, "request_id": [f"r{i // 2}" for i in ids]
            }).write_parquet(identity)
            probabilities = labels * 0.8 + 0.1
            first, second = root / "first.npz", root / "second.npz"
            np.savez(first, y_true=labels, probabilities=probabilities, row_ids=ids)
            np.savez(second, y_true=labels, probabilities=1 - probabilities, row_ids=ids)
            report = analyze(
                {"first": first, "second": second}, identity, root / "results",
                repetitions=10,
            )
            self.assertEqual(report["best_blend"]["validation_auc"], 1)
            self.assertFalse(report["competition_submission"])
            np.savez(second, y_true=labels, probabilities=probabilities, row_ids=ids[::-1])
            with self.assertRaises(AssertionError):
                analyze({"first": first, "second": second}, identity, root / "bad", repetitions=10)
            bootstrap = paired_group_bootstrap(
                labels, probabilities, probabilities,
                np.repeat(np.arange(8), 2), repetitions=10,
            )
            self.assertEqual(bootstrap["percentile_95_interval"], [0, 0])


if __name__ == "__main__":
    unittest.main()
