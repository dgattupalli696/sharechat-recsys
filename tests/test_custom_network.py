from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import polars as pl
import torch

from tabm_experiment import TabMClassifier, predict
from tabm_variants import AttentionTabMClassifier, main as tabm_main
from custom_network import (
    RecommendationNetwork,
    append_vectors,
    main,
    pack_vectors,
    prepare_custom_features,
    transform_custom_features,
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
        x_train, c_train, x_valid, c_valid, state = prepare_custom_features(train, valid, 3)
        self.assertEqual(state["cardinalities"], [2])
        self.assertTrue((c_valid.numpy() == 0).all())
        self.assertEqual(c_train[4, 0].item(), 0)
        self.assertGreater(c_train[0, 0].item(), 0)
        self.assertEqual(state["vector_fields"], ["feature26"])
        for original, encoded, data in [
            (x_train, c_train, train), (x_valid, c_valid, valid)
        ]:
            replay, categories = transform_custom_features(data, state)
            torch.testing.assert_close(replay, original)
            torch.testing.assert_close(categories, encoded)

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
        dense = torch.tensor([[1.0, 2.0]] * 3)
        appended, counts = append_vectors(pl.DataFrame(data), dense, ["feature26"])
        torch.testing.assert_close(appended, torch.cat([dense, packed.flatten(1)], dim=1))
        self.assertEqual(counts, missing)
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

    def test_attention_only_run_reads_an_explicit_raw_dataset_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "input"
            data.mkdir()
            n = 160
            expanded = frame(["common"] * n)
            raw = expanded.select("_time", "feature5", "feature33", "watched").rename(
                {"_time": "feature1"}
            ).with_columns(
                pl.Series(
                    "feature26",
                    [
                        None if i % 13 == 0 else [float((i + j) % 4) for j in range(32)]
                        for i in range(n)
                    ],
                )
            )
            raw.write_parquet(data / "train.parquet")
            output = root / "output"
            with patch.object(
                sys, "argv",
                [
                    "custom_network.py", "--rows", str(n), "--data-dir", str(data),
                    "--variant", "attention", "--epochs", "1", "--batch-size", "32",
                    "--threads", "1", "--device", "cpu", "--output-dir", str(output),
                ],
            ):
                main()
            result = json.loads((output / "results.json").read_text())
            self.assertEqual(result["training_rows"], 128)
            self.assertEqual([entry["variant"] for entry in result["results"]], ["attention"])
            self.assertTrue((output / "attention/best_model.pt").is_file())
            self.assertFalse((output / "control").exists())

    def test_tabm_variants_keep_the_full_holdout_when_sampling_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = root / "sample.parquet"
            data = frame(["common" if i % 3 else f"id-{i}" for i in range(160)])
            data.write_parquet(sample)
            states = []
            for training_rows in [None, 96]:
                output = root / str(training_rows)
                argv = [
                    "tabm_variants.py", "--sample-path", str(sample),
                    "--epochs", "1", "--batch-size", "32", "--members", "2",
                    "--width", "8", "--threads", "1", "--device", "cpu",
                    "--output-dir", str(output),
                ]
                if training_rows is not None:
                    argv += ["--training-rows", str(training_rows)]
                with patch.object(sys, "argv", argv):
                    tabm_main()
                report = json.loads((output / "results.json").read_text())
                states.append(report)
                self.assertEqual(report["training_rows"], training_rows or 128)
                self.assertEqual(report["validation_rows"], 32)
                self.assertFalse(report["competition_submission"])
                self.assertEqual(
                    [entry["variant"] for entry in report["results"]],
                    ["standalone", "hybrid"],
                )
                for variant, model_type in [
                    ("standalone", TabMClassifier), ("hybrid", AttentionTabMClassifier)
                ]:
                    path = output / variant
                    checkpoint = torch.load(path / "best_model.pt", weights_only=True)
                    model = model_type(**checkpoint["model_config"])
                    model.load_state_dict(checkpoint["state_dict"])
                    state = joblib.load(output / "preprocessing.joblib")
                    numeric, categories = transform_custom_features(data.slice(128), state)
                    actual = predict(model, numeric, categories, "cpu", 32)
                    saved = np.load(path / "validation_predictions.npz")
                    np.testing.assert_allclose(actual, saved["probabilities"], atol=1e-7)
                    np.testing.assert_array_equal(saved["y_true"], data["watched"][128:].to_numpy())
                    self.assertFalse((path / "submission.csv").exists())
            self.assertEqual(states[0]["cutoff"], states[1]["cutoff"])
            for key in ["validation_labels_sha256", "validation_timestamps_sha256"]:
                self.assertEqual(states[0][key], states[1][key])

    def test_updated_tabm_drops_categories_only_during_training(self):
        model = TabMClassifier(
            2, [3], members=2, width=8, dropout=0, category_dropout=1,
            positive_rate=0.05,
        )
        numeric = torch.zeros(4, 2)
        known = torch.ones(4, 1, dtype=torch.int64)
        unknown = torch.zeros_like(known)
        model.train()
        torch.testing.assert_close(model(numeric, known), model(numeric, unknown))
        captured = []
        handle = model.embeddings[0].register_forward_pre_hook(
            lambda module, inputs: captured.append(inputs[0].clone())
        )
        model.eval()
        model(numeric, known)
        handle.remove()
        torch.testing.assert_close(captured[0], known[:, 0])
        torch.testing.assert_close(
            model.backbone.output.bias,
            torch.full((2, 1), float(np.log(0.05 / 0.95))),
        )


if __name__ == "__main__":
    unittest.main()
