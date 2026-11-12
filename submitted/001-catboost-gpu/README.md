# S001 - Submitted initial GPU baseline

**Public leaderboard ROC AUC: 0.80699.**

| Field | Value |
| --- | --- |
| Submission ID | 58271142 |
| Submission time | November 12, 2026, 11:25:57.533 UTC |
| Kaggle kernel | `dheerajgattupalli/sharechat-catboost-gpu-baseline` |
| Executed kernel version | 2 |
| Submission file | `submission.csv` |
| Prediction rows | 1,037,090 |
| Model | Full-data 21-tree CatBoost GPU refit |
| Validation ROC AUC | 0.8160260803 |
| Submission status | Complete |

The user explicitly approved this one submission. **This approval does not
authorize another submission.** The notebook only trains and generates a CSV.

## Immutable archive

- `sharechat-catboost-gpu-baseline.py`: exact script downloaded from Kaggle version 2.
- `notebook.ipynb`: that source wrapped for notebook execution; model code unchanged.
- `kernel-metadata.json`: metadata exported with Kaggle version 2.
- `metrics.json`: original validation metrics and feature names.
- `leaderboard-submission.json`: original Kaggle grading record.
- `run.log`: original successful Kaggle execution log.
- `provenance.json`: source and prediction SHA-256 values and original file sizes.

The original CSV and binary models are deliberately excluded from Git; they are
saved locally in `artifacts/kaggle-v2/`. The private Kaggle run also retains its
outputs. Do not reconstruct or modify this directory for a different model:
create a new numbered submission archive only after approval.

The source's main guard, CLI defaults, validation, refit, and prediction logic
are preserved exactly. The notebook bootstrap sets `__file__` and `sys.argv`
because those differ from standalone script execution. Reruns write to
`artifacts/reruns/s001/` locally or `/kaggle/working/` on Kaggle and never submit.
