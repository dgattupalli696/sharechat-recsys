# S018 — T048 listwise-1.0 (holdout reliability check)

User approval: User input: "listwise1+finalmlp+xgbd12"

- Train-only checkpoint (fit rows before the original cutoff, no refit). Holdout 0.88420; expected public ≈0.8788 from S005's holdout→public gap.
- **Public 0.87973** (holdout − public 0.00447). Submission 58510176; file `artifacts/kaggle-t048/t048/listwise-1.0/prediction/submission.csv` (sha256 `923f947b0e1339aa`).
- Public order across train-only models (S018 < S019 < S005 < S020) matches holdout order.
