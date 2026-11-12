# T002 - Expand embedding dimensions versus summary statistics

Run date: November 12, 2026. Local-only experiment; neither model was submitted.
Open `notebook.ipynb` to reproduce the two frozen experiment implementations.

## Hypothesis

The fourteen list-valued columns each contain 32 values in the inspected data.
Keeping all 448 individual dimensions might preserve predictive information
lost by B001's vector summaries.

## Controlled comparison

- Same seed-42 uniform sample of 150,000 labeled rows in both variants.
- Same chronological holdout: 119,999 training and 30,001 validation rows.
- Cutoff: October 28, 2026, 08:06:29.680 in the dataset's timestamp representation.
- Same model: CatBoost CPU, four threads, depth 6, learning rate 0.06, maximum
  400 trees, Logloss/AUC, early stopping patience 60, `max_ctr_complexity=1`.
- T002-A: 679 summary-based features, matching the baseline feature strategy.
- T002-B: the same 679 features plus 448 vector dimensions, for 1,127 features.
  Non-null vectors with a length other than 32 are rejected.
- Hardware: Apple M1 Mac mini, 16 GB RAM. Sample preparation took 13.16 seconds.

## Results

| Variant | Validation AUC | Selected trees | Fit wall time |
| --- | ---: | ---: | ---: |
| T002-A: summaries | 0.8208754309 | 298 | 49.61 s |
| T002-B: expanded vectors | 0.8194781205 | 400 | 71.72 s |

Expansion changed validation AUC by -0.0013973104 and took about 45% longer.
It did not help under this configuration. T002-B reached the iteration cap, so
this does not rule out improvements with further tuning. There are no repeated
seeds or confidence intervals; this is one paired exploratory comparison.

These scores are not directly comparable to B001's full-data GPU holdout or its
public leaderboard score. GPU and CPU fits and sample sizes differ.

## Reproduction and saved outputs

The notebook freezes the shared feature/model definitions and experiment runner
used for this trial; it does not depend on future edits to those scripts.
`results.json` contains the original measured output. `provenance.json` identifies
the source snapshots by SHA-256.

The equivalent command at this source revision is:

```bash
uv run python experiments.py --rows 150000 --iterations 400 --threads 4 \
  --output-dir artifacts/local-v1
```

The cached sample and fitted models are local-only in `artifacts/local-v1/`.
Notebook reruns use `artifacts/reruns/t002/` so the original artifacts are not
overwritten. No Kaggle run or submission is launched by this notebook.
