# T001 - Timestamp quantile failure

Run date: November 12, 2026. Kaggle kernel version 1 failed before model fitting.
No competition submission was made from this version.

## Hypothesis and configuration

Attempt the initial full-data CatBoost GPU baseline with an 80/20 chronological
split based on `feature1`. The first implementation obtained its cutoff using
`sample["_time"].quantile(0.8, interpolation="nearest")`.

## Observed failure

In the Kaggle environment, the datetime quantile returned `None`. Comparing
timestamps to `None` produced null predicates, so both filtered partitions were
empty. The final error was `ValueError: train must contain both target classes`.
This was a code/environment compatibility failure, not a GPU cancellation.

`run.log` is the original version-1 execution log. `notebook.ipynb` demonstrates
the old and replacement cutoff expressions on synthetic timestamps; the old
expression's behavior can differ by Polars version.

## Fix and decision

Sort the rows by timestamp, take the timestamp at positional index
`int(row_count * 0.8)`, and split strictly before versus at/after that timestamp.
Keep timestamp ties together. Reject missing timestamps, fewer than two rows,
empty splits, or a partition without both classes.

The corrected version completed successfully as B001 / Kaggle version 2.
Regression checks covered a quantile returning `None`, tied timestamps, and an
all-identical-timestamp dataset. No metric is attributed to the failed run.
