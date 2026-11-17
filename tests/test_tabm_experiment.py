from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import polars as pl
import torch

from tabm_experiment import main, prepare_features


class TabMExperimentTests(unittest.TestCase):
    def test_preprocessing_learns_only_from_training_and_reserves_unknown_categories(self):
        train = pl.DataFrame(
            {
                "value": [1.0, None, 3.0, 5.0],
                "all_missing": pl.Series([None] * 4, dtype=pl.Float32),
                "category": ["a", "b", None, "a"],
                "watched": [0, 1, 0, 1],
            }
        )
        valid = pl.DataFrame(
            {
                "value": [999.0, None],
                "all_missing": [10.0, 20.0],
                "category": ["future-category", None],
                "watched": [1, 0],
            }
        )
        numeric, _, valid_numeric, valid_categories, state = prepare_features(train, valid)
        changed = valid.with_columns(pl.lit(-10000.0).alias("value"))
        numeric_again, _, _, _, state_again = prepare_features(train, changed)
        np.testing.assert_array_equal(numeric.numpy(), numeric_again.numpy())
        np.testing.assert_array_equal(state["scaler"].mean_, state_again["scaler"].mean_)
        self.assertEqual(state["imputer"].statistics_[0], 3.0)
        self.assertEqual(valid_categories[0, 0].item(), 0)
        self.assertGreater(valid_categories[1, 0].item(), 0)
        self.assertTrue(np.isfinite(valid_numeric.numpy()).all())
        self.assertLessEqual(np.abs(valid_numeric.numpy()).max(), 10)

    def test_validation_run_saves_reproducible_checkpoint_without_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = root / "sample.parquet"
            n = 160
            pl.DataFrame(
                {
                    "_time": [
                        datetime(2026, 11, 1) + timedelta(minutes=i // 4)
                        for i in range(n)
                    ],
                    "feature2": ["a" if i % 3 else None for i in range(n)],
                    "feature33": [float(i % 4) for i in range(n)],
                    "watched": [int(i % 4 >= 2) for i in range(n)],
                }
            ).write_parquet(sample)
            output = root / "output"
            argv = [
                "tabm_experiment.py",
                "--rows", str(n),
                "--sample-path", str(sample),
                "--epochs", "2",
                "--batch-size", "32",
                "--members", "2",
                "--width", "8",
                "--blocks", "2",
                "--threads", "1",
                "--device", "cpu",
                "--output-dir", str(output),
            ]
            with patch.object(sys, "argv", argv):
                main()
            results = json.loads((output / "results.json").read_text())
            self.assertEqual(results["training_rows"], 128)
            self.assertEqual(results["validation_rows"], 32)
            self.assertFalse(results["competition_submission"])
            self.assertTrue((output / "best_model.pt").is_file())
            self.assertTrue((output / "preprocessing.joblib").is_file())
            predictions = np.load(output / "validation_predictions.npz")
            self.assertEqual(len(predictions["y_true"]), 32)
            self.assertTrue(np.isfinite(predictions["probabilities"]).all())
            self.assertFalse((output / "submission.csv").exists())

    def test_cached_subset_cannot_silently_change_requested_row_count(self):
        with tempfile.TemporaryDirectory() as directory:
            sample = Path(directory) / "sample.parquet"
            pl.DataFrame({"value": [1, 2]}).write_parquet(sample)
            with patch.object(
                sys, "argv",
                [
                    "tabm_experiment.py", "--rows", "3",
                    "--sample-path", str(sample), "--device", "cpu",
                ],
            ):
                with self.assertRaisesRegex(ValueError, "row count"):
                    main()


if __name__ == "__main__":
    unittest.main()


class ListwiseTrainingTests(unittest.TestCase):
    def test_grouped_order_keeps_requests_contiguous_and_covers_every_row(self):
        from tabm_experiment import grouped_order

        groups = torch.tensor([5, 1, 5, 3, 1, 3, 5])
        order = grouped_order(groups, torch.Generator().manual_seed(0))
        assert sorted(order.tolist()) == list(range(7))
        seen = groups[order].tolist()
        runs = [g for i, g in enumerate(seen) if i == 0 or g != seen[i - 1]]
        assert len(runs) == 3


    def test_listwise_loss_is_softmax_cross_entropy_over_requests_with_a_positive(self):
        from tabm_experiment import listwise_loss

        logits = torch.tensor([[1.0], [2.0], [0.5], [3.0], [-1.0], [4.0]])
        targets = torch.tensor([[1.0], [0.0], [0.0], [0.0], [0.0], [1.0]])
        groups = torch.tensor([7, 7, 7, 9, 9, 11])  # group 9 has no positive, 11 has one row
        expected = -torch.log_softmax(torch.tensor([1.0, 2.0, 0.5]), 0)[0]
        assert torch.allclose(listwise_loss(logits, targets, groups), expected)
        assert listwise_loss(logits[3:5], targets[3:5], groups[3:5]).item() == 0
