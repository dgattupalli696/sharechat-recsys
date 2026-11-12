# B001 - Initial CatBoost GPU baseline

Run date: November 12, 2026. This is the reference configuration, not a local
150,000-row run. Open `notebook.ipynb` to reproduce the frozen submitted pipeline.

## Configuration

- Data: all 1,537,638 labeled rows, seed 42.
- Split: 1,230,110 training rows and 307,528 validation rows.
- Cutoff: October 28, 2026, 08:06:52.397 in the dataset's timestamp representation
  (timezone was not specified); timestamps equal to the cutoff stay in validation.
- Model: CatBoostClassifier, GPU, depth 6, learning rate 0.06, Logloss objective,
  AUC evaluation, maximum 400 trees, early stopping patience 60,
  `max_ctr_complexity=1`.
- Features: 679 total, including 23 categorical columns. Ignore `id` and
  `request_id`; derive hour and weekday from `feature1`; summarize each of
  fourteen 32-value vectors using length, mean, standard deviation, min, and max.
- Compute: Kaggle T4 accelerator configuration. The image digest is archived;
  exact Kaggle package versions were not enumerated.

## Results

| Measurement | Result |
| --- | ---: |
| Chronological validation ROC AUC | 0.8160260803 |
| Selected trees | 21 |
| Public leaderboard ROC AUC | 0.80699 |
| Test predictions | 1,037,090 |
| Full script runtime through prediction export | approximately 172 seconds |

The logged validation training loop stopped after approximately 5 seconds
(including patience iterations); the 21-tree full-data refit loop took 1.53
seconds. Those loop timers exclude substantial data preparation and model setup.
The whole pipeline took about three minutes.

After validation, refit 21 trees on all labeled rows. Test prediction is batched;
IDs, row counts, and finite probabilities in [0, 1] are checked.

## Reproduction and artifacts

`notebook.ipynb` contains the exact version-2 Python source in a code cell plus a
notebook bootstrap cell that supplies `__file__` and a clean argument list.
The bootstrap permits local execution with the repository's `data/` directory
or Kaggle execution with attached competition data. Local execution defaults to
a 150,000-row CPU verification run, not the original full-data GPU run; Kaggle
execution uses all rows. No submission is performed.

`metrics.json` and `provenance.json` record results and source identity. The
canonical submitted source, log, metadata, prediction checksum, and scoreboard
record are preserved in
[`submitted/001-catboost-gpu/`](../../submitted/001-catboost-gpu/README.md).

Large model and prediction files remain local in `artifacts/kaggle-v2/` and are
not committed. The initial failed attempt is tracked separately as T001.
