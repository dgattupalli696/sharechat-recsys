from pathlib import Path
import tempfile
import unittest

import numpy as np
import polars as pl

from request_features import ITEM, RANKED, augment


class RequestFeatureTests(unittest.TestCase):
    def test_augment_preserves_rows_and_computes_within_request_context(self):
        rng = np.random.default_rng(0)
        requests = ["a", "b", "a", "c", "b", "a"]
        frame = pl.DataFrame({
            "request_id": requests,
            ITEM: ["x", "y", "x", "z", "w", "v"],
            "watched": [0, 1, 1, 0, 0, 0],
            **{c: rng.normal(size=6).astype(np.float32) for c in RANKED},
        }).with_columns(pl.lit([1.0, 2.0]).alias("vector"))
        with tempfile.TemporaryDirectory() as directory:
            source, destination = Path(directory) / "s.parquet", Path(directory) / "d.parquet"
            frame.write_parquet(source)
            added = augment(source, destination, batch_size=2)
            result = pl.read_parquet(destination)
        self.assertTrue(result.select(frame.columns).equals(frame))
        self.assertEqual(result["rq_size"].to_list(), [3, 2, 3, 1, 2, 3])
        self.assertEqual(result["rq_items"].to_list(), [2, 2, 2, 1, 2, 2])
        self.assertEqual(result["rq_item_dup"].to_list(), [2, 1, 2, 1, 1, 1])
        values = frame[RANKED[0]].to_numpy()
        members = [0, 2, 5]
        np.testing.assert_allclose(
            result[f"rq_dev_{RANKED[0]}"].to_numpy()[members],
            values[members] - values[members].mean(), rtol=1e-6,
        )
        np.testing.assert_allclose(
            result[f"rq_gap_{RANKED[0]}"].to_numpy()[members],
            values[members] - values[members].max(), rtol=1e-6,
        )
        ranks = values[members].argsort().argsort() + 1
        np.testing.assert_allclose(result[f"rq_rank_{RANKED[0]}"].to_numpy()[members], ranks / 3)
        self.assertNotIn("watched", " ".join(added))
        self.assertTrue(all(result[c].dtype == pl.Float32 for c in added))


if __name__ == "__main__":
    unittest.main()
