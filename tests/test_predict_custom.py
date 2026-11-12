from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import polars as pl

from custom_network import main as train_main
from predict_custom import main as predict_main


class CustomPredictionTests(unittest.TestCase):
    def test_checkpoint_replay_prediction_alignment_and_environment_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            n = 160
            raw = pl.DataFrame(
                {
                    "feature1": [
                        datetime(2026, 11, 1) + timedelta(minutes=i // 4)
                        for i in range(n)
                    ],
                    "feature5": ["common" if i % 3 else f"id-{i}" for i in range(n)],
                    "feature33": [float(i % 4) for i in range(n)],
                    "feature26": [
                        [float((i + j) % 4) for j in range(32)] for i in range(n)
                    ],
                    "watched": [int(i % 4 >= 2) for i in range(n)],
                }
            )
            raw.write_parquet(data / "train.parquet")
            raw.head(10).drop("watched").with_columns(
                pl.int_range(0, pl.len()).alias("id")
            ).write_parquet(data / "test.parquet")
            template = pl.DataFrame({"id": range(10), "watched": [0.5] * 10})
            template.write_csv(data / "sample_submission.csv")
            model = root / "model"
            with patch.object(
                sys, "argv",
                [
                    "custom_network.py", "--data-dir", str(data), "--rows", str(n),
                    "--variant", "attention", "--epochs", "1", "--batch-size", "32",
                    "--threads", "1", "--device", "cpu", "--output-dir", str(model),
                ],
            ):
                train_main()

            def infer(output):
                with patch.object(
                    sys, "argv",
                    [
                        "predict_custom.py", "--model-dir", str(model),
                        "--data-dir", str(data), "--output-dir", str(output),
                        "--batch-size", "16", "--threads", "1", "--device", "cpu",
                    ],
                ):
                    predict_main()

            output = root / "predictions"
            infer(output)
            predictions = pl.read_csv(output / "submission.csv")
            self.assertTrue(predictions["id"].equals(template["id"]))
            self.assertTrue(predictions["watched"].is_between(0, 1).all())
            report = json.loads((output / "inference-results.json").read_text())
            self.assertAlmostEqual(
                report["replayed_validation_auc"], report["training_validation_auc"]
            )
            self.assertFalse(report["refit"])
            self.assertFalse(report["automatic_competition_submission"])

            template.with_columns(
                pl.Series("id", [1, 0, *range(2, 10)])
            ).write_csv(data / "sample_submission.csv")
            with self.assertRaisesRegex(ValueError, "IDs do not match"):
                infer(root / "misaligned")
            self.assertFalse((root / "misaligned/submission.csv").exists())

            training = json.loads((model / "results.json").read_text())
            training["versions"]["scikit-learn"] = "0.0.0"
            (model / "results.json").write_text(json.dumps(training))
            with self.assertRaisesRegex(RuntimeError, "versions differ"):
                infer(root / "wrong-environment")
            self.assertFalse((root / "wrong-environment/submission.csv").exists())


if __name__ == "__main__":
    unittest.main()
