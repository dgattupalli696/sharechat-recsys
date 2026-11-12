# T008 - Full-data TabM checkpoint, optimization and category controls

**Validation only. Every variant trains on all 1,230,110 training rows and
selects checkpoints on the same 307,528-row holdout. No test predictions or
competition submissions.**

## Protocol

Primary cutoff: October 28, 2026, 08:06:52.397 in dataset time, keeping ties
together. Read every labeled source row; do not sample the modeling population.
Store original row identities outside model inputs for paired evaluation.

Predeclare temporal confirmation before looking at new results: train strictly
before October 27, 2026, 08:06:52.397, validate from that instant until October 27,
2026, 18:00. This earlier backtest is a robustness check, not a retroactively
untouched test set.

Match T007-A: TabM eight members, two 128-unit blocks, backbone dropout 0.2,
eight-dimensional categorical embeddings with category dropout 0.3 and minimum
frequency 10, normalized vector direction/log magnitude/presence, train-only
numeric preprocessing, AdamW LR 0.0005 and weight decay 0.01, batch 1,024,
seed 42 and four CPU threads. Every model gets up to 30 epochs and five
epoch-equivalent nonimproving optimizer-step budgets.

The fine cadence is every 300 optimizer steps plus epoch end. Evaluation
restores training mode and does not change the optimizer trajectory. Save
the best global step/fractional epoch and verify restored AUC. A fixed
16,384-row training probe is diagnostic only: it is not the training population,
validation cohort, stopping criterion or a substitute for full-data experiments.

## Full-data runs

| Name | Single change from reference | Full validation AUC | Best fractional epoch |
| --- | --- | ---: | ---: |
| epoch-checkpoint-control | Original epoch-end cadence | 0.8292287153 | 1.0000 |
| reference | Fine checkpoint cadence | 0.8292340756 | 0.9983 |
| learning-rate-1e4 | LR 0.0001 | 0.8303946009 | 2.9950 |
| dropout-035 | Backbone dropout 0.35 | 0.8288247432 | 1.0000 |
| weight-decay-003 | AdamW weight decay 0.03 | 0.8290579579 | 0.9983 |
| width-64 | Two 64-unit blocks | 0.8283707922 | 1.0000 |
| numeric-dropout-01 | Continuous scalar dropout 0.1 | 0.8287533428 | 1.0000 |
| cosine-two-epochs | Decay LR to 10% over two epochs, then hold floor | 0.8308827050 | 1.7471 |
| minimum-category-50 | Training frequency threshold 50 | 0.8299448274 | 1.7471 |
| embedding-dim-4 | Four-dimensional category embeddings | 0.8290061527 | 1.0000 |
| adaptive-category-dims | Capped fourth-root dimensions | 0.8288613710 | 0.7488 |
| without-high-cardinality | Mask original training cardinality above 1,000 | **0.8334188580** | **7.7371** |

All twelve full-data runs completed. The epoch-only control reproduces T007-A
exactly, so the change in checkpoint cadence is isolated. Fine checking alone
improved AUC by only 0.0000053603. Masking the seven high-cardinality fields
improved the fine reference by 0.0041847823 and moved the best checkpoint well
past the first epoch. These fields are `feature5`, `feature9`, `feature12`,
`feature18`, `feature20`, `feature23`, and `feature24`; this joint ablation does
not identify each field's individual contribution.

The best variant trained for thirteen epochs before patience expired and
restored global step 9,300. Its fit/validation time was 252.96 seconds.
Preparation for the shared full dataset took 350.00 seconds. Every saved
prediction file independently reproduces its reported AUC and exactly matches
T007's held-out labels/order. No leaderboard score was generated.

No variant is eliminated using subset AUC. Record every completed outcome,
including regressions. Later experiments will test all planned encodings,
vector representations, tree models, DCNv2/FinalMLP, blending, temporal OOF
stacking and full-partition robustness checks.

## Tracking

Completed as version **3** of the existing
`dheerajgattupalli/sharechat-model-experiments-gpu` notebook. Frozen sources,
offline dependencies, configuration and source revision are tracked with
provenance. Keep large weights, fitted preprocessors and prediction arrays in
ignored `artifacts/kaggle-t008/`; archive small original reports/logs here.

Source revision: `8552afb306bd794bffe15b3906325580d8ddb1fc`.
Original metrics, log, protocol, exported metadata and artifact hashes are
archived alongside this README. No test prediction or competition submission.
