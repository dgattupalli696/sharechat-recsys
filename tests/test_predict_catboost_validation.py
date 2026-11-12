from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest

from catboost import CatBoostClassifier
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from predict_catboost_validation import CUTOFF, replay
from train import features


class BaselineReplayTests(unittest.TestCase):
    def test_saved_holdout_model_is_replayed_without_refitting_or_reordering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cutoff = datetime.fromisoformat(CUTOFF)
            raw = pl.DataFrame({
                "request_id": [f"r{i // 4}" for i in range(160)],
                "feature1": [cutoff + timedelta(seconds=i - 128) for i in range(160)],
                "feature5": ["a" if i % 2 else "b" for i in range(160)],
                "feature33": [float(i % 11) for i in range(160)],
                "watched": [i % 2 for i in range(160)],
            }).reverse()
            raw.write_parquet(root / "train.parquet")
            train = raw.filter(pl.col("feature1") < cutoff)
            valid = raw.with_row_index("_row_id").filter(pl.col("feature1") >= cutoff)
            model = CatBoostClassifier(iterations=2, depth=2, thread_count=1, verbose=False)
            model.fit(features(train).to_pandas(), train["watched"].to_numpy(), cat_features=["feature5"])
            model.save_model(str(root / "validation.cbm"))
            valid.select("_row_id", "request_id", "feature1").write_parquet(root / "identity.parquet")
            expected = model.predict_proba(features(valid.drop("_row_id")).to_pandas())[:, 1]
            report = replay(
                root / "train.parquet", root / "validation.cbm",
                root / "identity.parquet", root / "output",
                expected_auc=roc_auc_score(valid["watched"].to_numpy(), expected),
                threads=1,
            )
            arrays = np.load(root / "output/validation_predictions.npz")
            np.testing.assert_array_equal(arrays["probabilities"], expected)
            np.testing.assert_array_equal(arrays["row_ids"], valid["_row_id"].to_numpy())
            self.assertFalse(report["retraining"])
