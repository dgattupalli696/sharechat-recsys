from datetime import datetime, timedelta
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import polars as pl
import torch

from relational_audit import history_features
from relational_experiment import GraphTabM, METHODS, MatrixFactorization, graph_arrays, run_fold


class RelationalTests(unittest.TestCase):
    def test_history_excludes_current_timestamp_and_freezes_validation(self):
        time = datetime(2026, 10, 25)
        train = pl.DataFrame({
            "feature1": [time, time, time + timedelta(hours=1)],
            "a": ["x", "x", "x"], "b": ["y", "y", "z"], "watched": [1, 0, 1],
        })
        valid = pl.DataFrame({
            "feature1": [time + timedelta(days=1)] * 2,
            "a": ["x", "new"], "b": ["y", "new"], "watched": [0, 1],
        })
        first, heldout = history_features(train, valid, ["a", "b"])
        np.testing.assert_array_equal(first[:2, 0], [0, 0])
        np.testing.assert_allclose(first[:2, 1], [.05, .05])
        self.assertAlmostEqual(first[2, 0], np.log(3), places=6)
        self.assertAlmostEqual(first[2, 1], 2 / 22, places=6)
        self.assertAlmostEqual(heldout[0, 1], 3 / 23, places=6)
        np.testing.assert_allclose(heldout[1, [1, 4, 7]], [.05] * 3)
        changed = valid.with_columns((1 - pl.col("watched")).alias("watched"))
        np.testing.assert_array_equal(heldout, history_features(train, changed, ["a", "b"])[1])

    def test_graph_uses_unique_training_edges_and_excludes_missing_nodes(self):
        graph = graph_arrays(
            np.array([1, 1, 2, 0]), np.array([1, 1, 2, 2]),
            [3, 3], np.array([[1.], [3.], [7.], [9.]], dtype=np.float32), fanout=3,
        )
        self.assertEqual(graph["training_unique_edges"], 2)
        np.testing.assert_array_equal(graph["neighbors"][0], 0)
        np.testing.assert_array_equal(graph["neighbors"][1], 4)
        np.testing.assert_array_equal(graph["neighbors"][2], 5)
        self.assertEqual(graph["degree"][1], 1)
        self.assertEqual(graph["attributes"][1, 0], 2)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            graph_arrays(np.array([3]), np.array([1]), [3, 3], np.ones((1, 1)))

    def test_graph_model_handles_unseen_nodes_and_saves_exact_replay(self):
        config = {
            "encoder_config": {
                "n_numeric": 36, "n_dense": 2, "n_vectors": 1,
                "cardinalities": [3], "continuous_indices": [0, 1],
                "embedding_dimensions": [2], "numerical_encoding": "none",
            },
            "width": 8, "members": 2, "dropout": .1, "positive_rate": .25,
        }
        graph = graph_arrays(
            np.array([1, 1, 2]), np.array([1, 2, 2]), [3, 3],
            np.array([[1., 2.], [2., 3.], [4., 5.]], dtype=np.float32),
        )
        numeric = torch.randn(4, 36)
        categorical = torch.tensor([[1, 1, 1], [1, 0, 2], [0, 2, 0], [0, 0, 0]])
        for use_neighbors in [False, True]:
            model = GraphTabM(config, graph, [0, 1], use_neighbors=use_neighbors)
            output = model(numeric, categorical)
            self.assertEqual(output.shape, (4, 2))
            self.assertTrue(torch.isfinite(output).all())
            output.sum().backward()
            self.assertGreater(model.self_projection.weight.grad.abs().sum().item(), 0)
            if use_neighbors:
                self.assertGreater(model.neighbor_projection.weight.grad.abs().sum().item(), 0)
            model.eval()
            restored = GraphTabM(config, graph, [0, 1], use_neighbors=use_neighbors)
            restored.load_state_dict(model.state_dict(), strict=True)
            restored.eval()
            torch.testing.assert_close(model(numeric, categorical), restored(numeric, categorical))

    def test_matrix_factorization_unknown_nodes_use_only_prior_initially(self):
        model = MatrixFactorization([3, 4], .05)
        output = model(torch.zeros((2, 1)), torch.tensor([[0, 0], [0, 2]]))
        torch.testing.assert_close(output.sigmoid(), torch.full((2, 1), .05))
        self.assertEqual(output.shape, (2, 1))

    def test_complete_fold_runs_all_models_and_preserves_original_identities(self):
        cutoff = datetime(2026, 10, 28, 8, 6, 52, 397000)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = 160
            pl.DataFrame({
                "request_id": [f"request-{i}" for i in range(rows)],
                "feature1": [cutoff + timedelta(seconds=i - 128) for i in range(rows)],
                "feature5": [f"left-{i % 7}" for i in range(rows)],
                "feature18": [f"right-{i % 11}" for i in range(rows)],
                "feature33": [float(i % 30) for i in range(rows)],
                "feature26": [[float((i + j) % 4) for j in range(32)] for i in range(rows)],
                "watched": [i % 2 for i in range(rows)],
            }).write_parquet(root / "train.parquet")
            output = root / "output"
            results = run_fold(
                root / "train.parquet", output, ["feature5", "feature18"],
                "primary", epochs=1, device="cpu",
            )
            self.assertEqual([row["method"] for row in results], METHODS)
            for result in results:
                self.assertEqual(result["train_rows"], 128)
                self.assertEqual(result["validation_rows"], 32)
                self.assertEqual(result["optimizer_steps"], 1)
                with np.load(output / result["method"] / "validation_predictions.npz") as arrays:
                    np.testing.assert_array_equal(arrays["row_ids"], np.arange(128, 160))
                    self.assertTrue(np.isfinite(arrays["probabilities"]).all())
            protocol = json.loads((output / "relational-protocol.json").read_text())
            self.assertFalse(protocol["graph_uses_validation_rows"])
            self.assertFalse(protocol["graph_uses_labels"])
            self.assertFalse(protocol["competition_submission"])


if __name__ == "__main__":
    unittest.main()
