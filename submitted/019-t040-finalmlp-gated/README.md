# S019 — T040 FinalMLP-gated (holdout reliability check)

User approval: User input: "listwise1+finalmlp+xgbd12"

- Train-only checkpoint (fit rows before the original cutoff, no refit). Holdout 0.88854; expected public ≈0.8831 from S005's holdout→public gap.
- **Public 0.88377** (holdout − public 0.00477). Submission 58510194; file `artifacts/kaggle-t040/t040/finalmlp-gated/prediction/submission.csv` (sha256 `e1925522369e00bb`).
- Public order across train-only models (S018 < S019 < S005 < S020) matches holdout order.
