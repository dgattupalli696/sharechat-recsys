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

from custom_network import (
    RecommendationNetwork,
    main,
    pack_vectors,
    prepare_custom_features,
)


def frame(categories):
    n = len(categories)
    columns = {
        "_time": [
            datetime(2026, 11, 1) + timedelta(minutes=i // 4) for i in range(n)
        ],
        "feature5": categories,
        "feature33": [float(i % 4) for i in range(n)],
        "watched": [int(i % 4 >= 2) for i in range(n)],
    }
    columns.update(
        {
            f"feature26_dim_{index}": [
                float((i + index) % 4) for i in range(n)
            ]
            for index in range(32)
        }
    )
    return pl.DataFrame(columns)


class CustomNetworkTests(unittest.TestCase):
    def test_rare_categories_are_retained_using_only_training_counts(self):
        train = frame(["common"] * 4 + ["rare", "another-rare", "common", "common"])
        valid = frame(["rare"] * 5 + ["unseen"] * 3)
        _, c_train, _, c_valid, state = prepare_custom_features(train, valid, 3)
        self.assertEqual(state["cardinalities"], [2])
        self.assertTrue((c_valid.numpy() == 0).all())
        self.assertEqual(c_train[4, 0].item(), 0)
        self.assertGreater(c_train[0, 0].item(), 0)
        self.assertEqual(state["vector_fields"], ["feature26"])

    def test_vector_packing_preserves_direction_and_handles_missing_fields(self):
        data = {
            f"feature26_dim_{index}": [
                3.0 if index == 0 else 4.0 if index == 1 else 0.0,
                None,
                0.0,
            ]
            for index in range(32)
        }
        packed, missing = pack_vectors(pl.DataFrame(data), ["feature26"])
        np.testing.assert_allclose(packed[0, 0, :2].numpy(), [0.6, 0.8])
        self.assertAlmostEqual(packed[0, 0, 32].item(), np.log1p(5), places=6)
        self.assertEqual(packed[:, 0, 33].tolist(), [1.0, 0.0, 1.0])
        self.assertEqual(missing["feature26"], 1)
        self.assertTrue(torch.isfinite(packed).all().item())
        for attention in [False, True]:
            model = RecommendationNetwork(2, [2], 1, attention=attention)
            numeric = torch.cat([torch.zeros(3, 2), packed.flatten(1)], dim=1)
            logits = model(numeric, torch.zeros(3, 1, dtype=torch.int64))
            logits.sum().backward()
            self.assertTrue(torch.isfinite(logits).all().item())
            self.assertTrue(
                all(
                    torch.isfinite(parameter.grad).all().item()
                    for parameter in model.parameters()
                    if parameter.grad is not None
                )
            )

    def test_paired_trial_saves_both_checkpoints_on_the_same_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = root / "sample.parquet"
            frame(["common" if i % 3 else f"id-{i}" for i in range(160)]).write_parquet(sample)
            output = root / "output"
            argv = [
                "custom_network.py", "--rows", "160", "--sample-path", str(sample),
                "--epochs", "2", "--batch-size", "32", "--threads", "1",
                "--device", "cpu", "--output-dir", str(output),
            ]
            with patch.object(sys, "argv", argv):
                main()
            result = json.loads((output / "results.json").read_text())
            self.assertEqual(result["training_rows"], 128)
            self.assertEqual(result["validation_rows"], 32)
            self.assertEqual(
                [entry["variant"] for entry in result["results"]],
                ["control", "attention"],
            )
            labels = []
            for variant in ["control", "attention"]:
                self.assertTrue((output / variant / "best_model.pt").is_file())
                pred = np.load(output / variant / "validation_predictions.npz")
                labels.append(pred["y_true"])
                self.assertTrue(np.isfinite(pred["probabilities"]).all())
                self.assertFalse((output / variant / "submission.csv").exists())
            np.testing.assert_array_equal(*labels)
            self.assertFalse(result["competition_submission"])


if __name__ == "__main__":
    unittest.main()
