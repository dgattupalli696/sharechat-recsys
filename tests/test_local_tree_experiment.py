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
