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

| Name | Single change from reference |
| --- | --- |
| epoch-checkpoint-control | Original epoch-end cadence; checks full-data reference reproduction |
| reference | Fine checkpoint cadence; all other T007-A settings unchanged |
| learning-rate-1e4 | LR 0.0001 |
| dropout-035 | Backbone dropout 0.35 |
| weight-decay-003 | AdamW weight decay 0.03 |
| width-64 | Two 64-unit blocks |
| numeric-dropout-01 | Drop eligible continuous scalar inputs with probability 0.1 |
| cosine-two-epochs | Decay initial LR to 10% over two epochs, then hold the floor |
| minimum-category-50 | Retain only categories occurring at least 50 times in training |
| embedding-dim-4 | Four-dimensional categorical embeddings |
| adaptive-category-dims | `min(32, ceil(cardinality ** 0.25))` per field |
| without-high-cardinality | Mask fields with original training cardinality above 1,000 |

No variant is eliminated using subset AUC. Record every completed outcome,
including regressions. Later experiments will test all planned encodings,
vector representations, tree models, DCNv2/FinalMLP, blending, temporal OOF
stacking and full-partition robustness checks.

## Tracking

Pending execution in the next version of the existing
`dheerajgattupalli/sharechat-model-experiments-gpu` notebook. Frozen sources,
offline dependencies, configuration and source revision are tracked with
provenance. Keep large weights, fitted preprocessors and prediction arrays in
ignored `artifacts/kaggle-t008/`; archive small original reports/logs here.
