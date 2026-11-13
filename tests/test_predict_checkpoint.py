from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest

import numpy as np
import polars as pl

from full_experiments import PRIMARY_CUTOFF, prepare_data, run_tabm
from predict_checkpoint import run
from request_features import augment


class PredictCheckpointTests(unittest.TestCase):
    def test_replays_holdout_and_writes_template_ordered_submission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cutoff = datetime.fromisoformat(PRIMARY_CUTOFF)
            rows = 160
            frame = pl.DataFrame({
                "request_id": [f"request-{i // 4}" for i in range(rows)],
                "feature1": [cutoff + timedelta(seconds=i - 128) for i in range(rows)],
                "feature5": ["common" if i % 3 else f"id-{i}" for i in range(rows)],
                "feature18": [f"item-{i % 5}" for i in range(rows)],
                "feature33": [float(i % 30) for i in range(rows)],
                "feature26": [[float((i + j) % 4) for j in range(32)] for i in range(rows)],
                "watched": [i % 2 for i in range(rows)],
            })
            raw = root / "raw"
            raw.mkdir()
            frame.write_parquet(raw / "train.parquet")
            test = frame.head(24).drop("watched").with_columns(
                pl.Series("id", np.arange(500, 524)[::-1].copy()),
                pl.Series("request_id", [f"test-{i // 3}" for i in range(24)]),
            )
            test.write_parquet(raw / "test.parquet")
            data_dir = root / "data"
            for name in ["train", "test"]:
                augment(raw / f"{name}.parquet", data_dir / f"{name}.parquet",
                        ranked=["feature33"], stats=["feature33"])
            pl.DataFrame({"id": test["id"], "watched": [0.] * 24}).write_csv(data_dir / "sample_submission.csv")
            output = root / "output"
            output.mkdir()
            data = prepare_data(data_dir / "train.parquet", output)
            run_tabm(data, {"name": "k", "width": 8, "members": 2, "drop_high": True,
                      "vector_crosses": True, "numerical_encoding": "plr"},
                     output, device="cpu", epochs=1, interval=1)
            report = run(output / "k", output / "preprocessing.joblib", data_dir, root / "prediction",
                         cutoff, device="cpu")
            self.assertEqual(report["holdout_rows"], 32)
            self.assertLessEqual(report["max_replay_error"], 1e-6)
            submission = pl.read_csv(root / "prediction/submission.csv")
            self.assertEqual(submission["id"].to_list(), test["id"].to_list())
            self.assertTrue(((submission["watched"] >= 0) & (submission["watched"] <= 1)).all())


if __name__ == "__main__":
    unittest.main()
