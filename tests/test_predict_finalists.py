from datetime import datetime, timedelta
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import polars as pl

from full_experiments import PRIMARY_CUTOFF, prepare_data, run_tabm
from local_tree_experiment import run as run_tree
from predict_finalists import combine, digest, run
from train import features


class FinalistPredictionTests(unittest.TestCase):
    def test_exact_checkpoint_replay_and_two_ordered_exports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, prepared, bundle = [root / name for name in ["data", "prepared", "bundle"]]
            for path in [data, prepared, bundle]:
                path.mkdir()
            cutoff = datetime.fromisoformat(PRIMARY_CUTOFF)
            raw = pl.DataFrame({
                "request_id": [f"r{i // 4}" for i in range(160)],
                "feature1": [cutoff + timedelta(seconds=i - 128) for i in range(160)],
                "feature5": ["common"] * 160,
                "feature33": [float(i % 30) for i in range(160)],
                "feature26": np.random.default_rng(42).normal(size=(160, 32)).tolist(),
                "watched": [i % 2 for i in range(160)],
            }).reverse()
            raw.write_parquet(data / "train.parquet")
            test = raw.head(19).drop("watched").with_columns(pl.Series("id", range(100, 119)))
            test.write_parquet(data / "test.parquet")
            pl.DataFrame({"id": range(100, 119), "watched": [0.0] * 19}).write_csv(data / "sample_submission.csv")
            training = prepare_data(data / "train.parquet", prepared)
            references, records = {}, {}
            for name, changes in [
                ("tabm32", {"members": 4}), ("tabm8", {"members": 2}),
                ("dcn32", {"architecture": "dcnv2", "rank": 4}),
            ]:
                report = run_tabm(
                    training, {"name": name, "width": 8, "minimum": 1000,
                               "numerical_encoding": "plr", **changes},
                    prepared, device="cpu", epochs=1,
                )
                shutil.copyfile(prepared / name / "best_model.pt", bundle / f"{name}.pt")
                records[name] = {"checkpoint": f"{name}.pt", "model_config": report["model_config"]}
                with np.load(prepared / name / "validation_predictions.npz") as predictions:
                    references[name] = predictions["probabilities"].copy()
            tree = root / "tree"
            run_tree(
                data / "train.parquet", tree, iterations=2, threads=1, fixed_budget=True,
                reference_identity=prepared / "validation-identities.parquet",
            )
            for source, target in [
                (prepared / "preprocessing.joblib", "preprocessing.joblib"),
                (prepared / "tabm32/category-remappings.joblib", "category-remappings.joblib"),
                (tree / "lightgbm/model.txt", "lightgbm.txt"),
                (tree / "category-vocabularies.json", "category-vocabularies.json"),
            ]:
                shutil.copyfile(source, bundle / target)
            with np.load(tree / "lightgbm/validation_predictions.npz") as predictions:
                references["lightgbm"] = predictions["probabilities"].copy()
            weights = {"tabm32": 0.45, "tabm8": 0.225, "dcn32": 0.225, "lightgbm": 0.1}
            references["ensemble"] = combine(references, weights)
            np.savez(
                bundle / "validation-reference.npz", y_true=training["y_valid"],
                row_ids=training["validation_row_ids"], **references,
            )
            manifest = {
                "files_sha256": {path.name: digest(path) for path in bundle.iterdir()},
                "versions": training["protocol"]["versions"],
                "neural_models": records, "cutoff": PRIMARY_CUTOFF, "validation_rows": 32,
                "tree_columns": features(raw.head(1), expand_embeddings=True).columns,
                "ensemble_weights": weights,
            }
            (bundle / "manifest.json").write_text(json.dumps(manifest))
            report = run(bundle, data, root / "predicted", device="cpu")
            for name in ["tabm", "ensemble"]:
                exported = pl.read_csv(root / "predicted" / report["submissions"][name]["file"])
                self.assertEqual(exported["id"].to_list(), list(range(100, 119)))
                self.assertEqual(exported.columns, ["id", "watched"])
            self.assertFalse(report["refit"])
            self.assertFalse(report["automatic_competition_submission"])
            self.assertFalse(list((root / "predicted").glob("finalist-inference-*")))
            (bundle / "category-vocabularies.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                run(bundle, data, root / "invalid", device="cpu")
