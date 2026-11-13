"""Label-free within-request context features appended to raw ShareChat parquet files.

Every row of a request shares one timestamp and requests never cross the
train/holdout/test boundaries, so these features are computed per request and
use no labels. request_id itself is never a model input.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq


ITEM = "feature18"
RANKED = ["feature418", "feature419", "feature411", "feature410", "feature430",
          "feature431", "feature417", "feature352", "feature346", "feature480"]


def request_expressions(ranked=RANKED, stats=()):
    group = "request_id"
    size = pl.len().over(group)
    exprs = [size.alias("rq_size"),
             pl.col(ITEM).n_unique().over(group).alias("rq_items"),
             pl.len().over([group, ITEM]).alias("rq_item_dup")]
    for c in ranked:
        exprs += [
            (pl.col(c).rank("average").over(group) / size).alias(f"rq_rank_{c}"),
            (pl.col(c) - pl.col(c).mean().over(group)).alias(f"rq_dev_{c}"),
            (pl.col(c) - pl.col(c).max().over(group)).alias(f"rq_gap_{c}"),
        ]
    for c in stats:
        exprs += [pl.col(c).mean().over(group).alias(f"rq_mean_{c}"),
                  pl.col(c).std().over(group).alias(f"rq_std_{c}")]
    return [e.cast(pl.Float32) for e in exprs]


def request_features(frame, ranked=RANKED, stats=()):
    """Return request features in the same row order as ``frame``."""
    return frame.select(request_expressions(ranked, stats))


def augment(source, destination, batch_size=65536, ranked=RANKED, stats=()):
    """Copy ``source`` to ``destination`` with request features appended, preserving row order."""
    narrow = pl.read_parquet(source, columns=list(dict.fromkeys(["request_id", ITEM, *ranked, *stats])))
    extra = request_features(narrow, ranked, stats)
    del narrow
    parquet = pq.ParquetFile(source)
    if parquet.metadata.num_rows != len(extra):
        raise ValueError("Request features do not cover every source row")
    overlap = set(extra.columns) & set(parquet.schema_arrow.names)
    if overlap:
        raise ValueError(f"Source already contains request features: {sorted(overlap)}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer, offset = None, 0
    try:
        for batch in parquet.iter_batches(batch_size=batch_size):
            table = pa.Table.from_batches([batch])
            appended = extra.slice(offset, len(table)).to_arrow()
            for name in appended.column_names:
                table = table.append_column(name, appended[name])
            if writer is None:
                writer = pq.ParquetWriter(destination, table.schema)
            writer.write_table(table)
            offset += len(table)
    finally:
        if writer is not None:
            writer.close()
    if offset != len(extra):
        raise ValueError("Augmented parquet does not cover every source row")
    return extra.columns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ranked-json", type=Path, help="JSON list of fields to rank within requests")
    args = parser.parse_args()
    for name in ["train", "test"]:
        ranked = json.loads(args.ranked_json.read_text()) if args.ranked_json else RANKED
        columns = augment(args.data_dir / f"{name}.parquet", args.output_dir / f"{name}.parquet", ranked=ranked)
        print(f"{name}: appended {len(columns)} request features", flush=True)
    sample = args.data_dir / "sample_submission.csv"
    target = args.output_dir / "sample_submission.csv"
    if sample.is_file() and not target.exists():
        target.symlink_to(sample.resolve())


if __name__ == "__main__":
    main()
