# S006 — Request-context ensemble

User approval: "can we try submissiong with tabm k32 and ensemble?"

- Models: unchanged T024 checkpoints, probability-weighted with T016 weights (0.45 k32, 0.225 k8, 0.225 DCNv2-r32, 0.10 LightGBM); no refit.
- Original holdout AUC: 0.889818 (replayed on Kaggle, max probability error 6.0e-8).
- Shared inference run with [S005](../005-request-tabm-k32/README.md).
- Submission 58300782: **public 0.88438**, our best so far (S004: 0.83365).
