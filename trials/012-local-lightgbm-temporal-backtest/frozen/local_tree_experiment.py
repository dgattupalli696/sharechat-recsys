import argparse
from datetime import datetime
import gc
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from sklearn.metrics import log_loss, roc_auc_score

from train import ROOT, TARGET, TIME, features


CUTOFF = "2026-10-28T08:06:52.397000"


def prepare_arrays(
    path, cache, output, reference_identity=None, *,
    cutoff_string=CUTOFF, validation_end=None,
):
    cutoff = datetime.fromisoformat(cutoff_string)
    parquet = pq.ParquetFile(path)
    raw_identity = pl.read_parquet(path, columns=[TIME, TARGET, "request_id"]).with_row_index("_row_id")
    identity = raw_identity.sort(TIME)
    training = identity.filter(pl.col(TIME) < cutoff)
    validation_mask = pl.col(TIME) >= cutoff
    if validation_end is not None:
        validation_mask &= pl.col(TIME) < datetime.fromisoformat(validation_end)
    validation = identity.filter(validation_mask)
    if not len(training) or not len(validation):
        raise ValueError("Both full temporal partitions must be nonempty")
    if reference_identity is not None:
        expected = pl.read_parquet(reference_identity)
        np.testing.assert_array_equal(
            np.sort(validation["_row_id"].to_numpy()), np.sort(expected["_row_id"].to_numpy())
        )
        validation = raw_identity[expected["_row_id"].to_numpy()]
        if not validation.select("_row_id", "request_id", TIME).equals(
            expected.select("_row_id", "request_id", TIME)
        ):
            raise ValueError("Reference identities do not match the source data")
    for frame in [training, validation]:
        if set(frame[TARGET].unique().to_list()) != {0, 1}:
            raise ValueError("Each temporal partition requires both target classes")
    rows = parquet.metadata.num_rows
    if rows == 1537638 and cutoff_string == CUTOFF and validation_end is None:
        assert len(training) == 1230110 and len(validation) == 307528
    training_ids, validation_ids = training["_row_id"].to_numpy(), validation["_row_id"].to_numpy()
    validation.select("_row_id", "request_id", TIME).write_parquet(output / "validation-identities.parquet")
    first = features(pl.from_arrow(next(parquet.iter_batches(batch_size=1))), expand_embeddings=True)
    columns = first.columns
    categories = [name for name, dtype in first.schema.items() if dtype == pl.String]
    del first
    category_frame = (
        pl.scan_parquet(path).select(TIME, *categories)
        .filter(pl.col(TIME) < cutoff).sort(TIME).collect()
    )
    vocabularies = {
        name: category_frame[name].fill_null("__MISSING__").unique(maintain_order=True).to_list()
        for name in categories
    }
    if any(len(values) >= 2**24 for values in vocabularies.values()):
        raise ValueError("Categorical codes exceed exact float32 integer capacity")
    del category_frame
    mappings = {name: {value: i for i, value in enumerate(values)} for name, values in vocabularies.items()}
    (output / "category-vocabularies.json").write_text(json.dumps(vocabularies) + "\n")
    del vocabularies
    category_expressions = [
        pl.col(name).replace_strict(mapping, default=-1, return_dtype=pl.Int32)
        for name, mapping in mappings.items()
    ]
    del mappings
    positions = {}
    arrays = {}
    for name, indices in [("train", training_ids), ("valid", validation_ids)]:
        positions[name] = np.full(rows, -1, dtype=np.int64)
        positions[name][indices] = np.arange(len(indices))
        arrays[name] = np.lib.format.open_memmap(
            cache / f"{name}.npy", mode="w+", dtype=np.float32,
            shape=(len(indices), len(columns)),
        )
    written = {"train": 0, "valid": 0}
    offset = 0
    for batch in parquet.iter_batches(batch_size=8192):
        frame = features(pl.from_arrow(batch), expand_embeddings=True)
        frame = frame.with_columns(category_expressions)
        matrix = frame.select(pl.all().cast(pl.Float32)).to_numpy()
        for name in ["train", "valid"]:
            destination = positions[name][offset : offset + len(frame)]
            keep = destination >= 0
            arrays[name][destination[keep]] = matrix[keep]
            written[name] += int(keep.sum())
        offset += len(frame)
    if offset != rows or written != {"train": len(training), "valid": len(validation)}:
        raise ValueError("Streaming preprocessing did not cover the complete dataset")
    for array in arrays.values():
        array.flush()
    np.savez(
        cache / "labels.npz", train=training[TARGET].to_numpy(),
        valid=validation[TARGET].to_numpy(), row_ids=validation_ids,
    )
    (cache / "metadata.json").write_text(json.dumps({
        "columns": columns, "categories": categories,
    }) + "\n")
    protocol = {
        "source_rows": rows, "training_rows": len(training), "validation_rows": len(validation),
        "cutoff": cutoff_string, "validation_end": validation_end,
        "input_features": len(columns),
        "validation_labels_sha256": hashlib.sha256(
            validation[TARGET].to_numpy().astype(np.uint8).tobytes()
        ).hexdigest(),
        "scope": "all rows in both full temporal partitions; no training subset",
        "platform": platform.platform(), "competition_submission": False,
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    return protocol


def run(
    path, output, *, iterations=1000, threads=2, reference_identity=None,
    cutoff_string=CUTOFF, validation_end=None,
):
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    required = pq.ParquetFile(path).metadata.num_rows * 1127 * 4
    if shutil.disk_usage(output).free < required + 2 * 1024**3:
        raise OSError("Insufficient free disk space for the full disk-backed feature matrices")
    with tempfile.TemporaryDirectory(prefix="streaming-lightgbm-", dir=output) as directory:
        cache = Path(directory)
        protocol = prepare_arrays(
            path, cache, output, reference_identity,
            cutoff_string=cutoff_string, validation_end=validation_end,
        )
        preparation_seconds = time.monotonic() - started
        print(f"FULL DATA READY: {json.dumps(protocol)}", flush=True)
        gc.collect()
        destination = output / "lightgbm"
        destination.mkdir(exist_ok=True)
        fit_started = time.monotonic()
        subprocess.run([
            sys.executable, str(Path(__file__).with_name("lightgbm_worker.py")),
            "--array-dir", str(cache), "--output-dir", str(destination),
            "--iterations", str(iterations), "--threads", str(threads),
        ], check=True)
        predictions = np.load(destination / "validation_predictions.npz")
        probabilities = predictions["probabilities"]
        if not np.isfinite(probabilities).all():
            raise ValueError("Nonfinite full-data LightGBM predictions")
        worker = json.loads((destination / "worker-results.json").read_text())
        result = {
            "name": "lightgbm", **worker,
            "validation_auc": float(roc_auc_score(predictions["y_true"], probabilities)),
            "validation_logloss": float(log_loss(predictions["y_true"], probabilities, labels=[0, 1])),
            "training_and_prediction_seconds": time.monotonic() - fit_started,
            "preparation_seconds": preparation_seconds,
        }
        (destination / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        report = {
            "stage": "local-trees", "status": "COMPLETE", "protocol": protocol,
            "results": [result], "competition_submission": False,
        }
        (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--reference-identities", type=Path)
    parser.add_argument("--cutoff", default=CUTOFF)
    parser.add_argument("--validation-end")
    args = parser.parse_args()
    if min(args.iterations, args.threads) <= 0:
        parser.error("Iterations and CPU threads must be positive")
    run(
        args.data_dir / "train.parquet", args.output_dir,
        iterations=args.iterations, threads=args.threads,
        reference_identity=args.reference_identities,
        cutoff_string=args.cutoff, validation_end=args.validation_end,
    )


if __name__ == "__main__":
    main()
