import argparse
from datetime import datetime
import itertools
import json
from pathlib import Path

import numpy as np
import polars as pl


PRIMARY = datetime.fromisoformat("2026-10-28T08:06:52.397000")
EARLIER = datetime.fromisoformat("2026-10-27T08:06:52.397000")
EARLIER_END = datetime.fromisoformat("2026-10-27T18:00:00")


def audit(path, output):
    output.mkdir(parents=True, exist_ok=True)
    lazy = pl.scan_parquet(path)
    schema = lazy.collect_schema()
    fields = [name for name, dtype in schema.items() if dtype == pl.String and name != "request_id"]
    frame = lazy.select("feature1", *fields).collect()
    train = frame.filter(pl.col("feature1") < PRIMARY)
    valid = frame.filter(pl.col("feature1") >= PRIMARY)
    selection_train = frame.filter(pl.col("feature1") < EARLIER)
    summary = []
    for field in fields:
        counts = train.filter(pl.col(field).is_not_null()).group_by(field).len()
        joined = valid.select(field).join(counts, on=field, how="left")
        repeated = int(counts.filter(pl.col("len") > 1)["len"].sum())
        record = {
            "field": field, "training_unique": len(counts),
            "training_nonmissing": int(counts["len"].sum()),
            "training_repeat_row_fraction": repeated / len(train),
            "training_max_count": int(counts["len"].max()),
            "training_median_count": float(counts["len"].median()),
            "training_missing_fraction": train[field].null_count() / len(train),
            "validation_seen_fraction": joined["len"].is_not_null().mean(),
        }
        selection_counts = selection_train.filter(pl.col(field).is_not_null()).group_by(field).len()
        record.update({
            "selection_training_unique": len(selection_counts),
            "selection_training_repeat_row_fraction": (
                int(selection_counts.filter(pl.col("len") > 1)["len"].sum()) / len(selection_train)
            ),
            "selection_training_missing_fraction": selection_train[field].null_count() / len(selection_train),
        })
        summary.append(record)
    # Freeze candidates using training structure only, never held-out labels/coverage.
    candidates = [
        row for row in summary
        if 128 <= row["selection_training_unique"] <= len(selection_train) // 2
        and row["selection_training_repeat_row_fraction"] >= .5
        and row["selection_training_missing_fraction"] < .2
    ]
    candidates.sort(key=lambda row: (-row["selection_training_unique"], row["field"]))
    pairs = []
    for left, right in itertools.combinations([row["field"] for row in candidates[:6]], 2):
        complete = selection_train.select(left, right).drop_nulls()
        edges = complete.unique()
        left_degree = edges.group_by(left).len()
        right_degree = edges.group_by(right).len()
        record = {
            "fields": [left, right], "training_edges": len(edges),
            "left_nodes": left_degree.height, "right_nodes": right_degree.height,
            "left_multineighbor_fraction": float((left_degree["len"] > 1).mean()),
            "right_multineighbor_fraction": float((right_degree["len"] > 1).mean()),
            "repeated_pair_row_fraction": 1 - len(edges) / max(1, complete.height),
        }
        record["structural_score"] = float(
            min(record["left_nodes"], record["right_nodes"])
            * record["left_multineighbor_fraction"] * record["right_multineighbor_fraction"]
        )
        pairs.append(record)
    pairs.sort(key=lambda row: (-row["structural_score"], row["fields"]))
    selected = pairs[0]["fields"] if pairs and pairs[0]["structural_score"] > 0 else None
    report = {
        "training_rows": len(train), "validation_rows": len(valid),
        "fields": summary, "candidate_pairs": pairs, "selected_fields": selected,
        "selection_training_cutoff": EARLIER.isoformat(),
        "selection_training_rows": len(selection_train),
        "selection_uses": "training cardinality, repeated coverage and many-to-many connectivity only",
        "semantics": "anonymous entity-like categories; not confirmed user/item identifiers",
        "competition_submission": False,
    }
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


def history_features(train, valid, fields, prior=.05, strength=20.0):
    """Exclude the current timestamp group, including all tied labels."""
    outputs = [[], []]
    for keys in [[field] for field in fields] + [list(fields)]:
        groups = (
            train.group_by([*keys, "feature1"])
            .agg(pl.len().alias("_count"), pl.col("watched").sum().alias("_positive"))
            .sort([*keys, "feature1"])
            .with_columns(
                (pl.col("_count").cum_sum().over(keys) - pl.col("_count")).alias("_past_count"),
                (pl.col("_positive").cum_sum().over(keys) - pl.col("_positive")).alias("_past_positive"),
            )
        )
        totals = train.group_by(keys).agg(
            pl.len().alias("_past_count"), pl.col("watched").sum().alias("_past_positive"),
        )
        for index, (frame, table, join_keys) in enumerate([
            (train, groups, [*keys, "feature1"]), (valid, totals, keys),
        ]):
            joined = frame.select(join_keys).with_row_index("_order").join(
                table.select(*join_keys, "_past_count", "_past_positive"),
                on=join_keys, how="left", nulls_equal=True,
            ).sort("_order").with_columns(
                pl.col("_past_count", "_past_positive").fill_null(0).cast(pl.Float64),
            )
            values = joined.select(
                pl.col("_past_count").log1p(),
                ((pl.col("_past_positive") + prior * strength)
                 / (pl.col("_past_count") + strength)).alias("_rate"),
                (pl.col("_past_count") > 0).cast(pl.Float64).alias("_seen"),
            ).to_numpy().astype(np.float32)
            if not np.isfinite(values).all():
                raise ValueError("Nonfinite history features")
            outputs[index].append(values)
    return tuple(np.concatenate(parts, axis=1) for parts in outputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/train.parquet"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.input, args.output)


if __name__ == "__main__":
    main()
