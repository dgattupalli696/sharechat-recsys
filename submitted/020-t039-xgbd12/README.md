# S020 — T039 XGBoost depth12 (holdout reliability check)

User approval: User input: "listwise1+finalmlp+xgbd12"

- Train-only checkpoint (fit rows before the original cutoff, no refit). Holdout 0.88967; expected public ≈0.8843 from S005's holdout→public gap.
- **Public 0.88482** (holdout − public 0.00485). Submission 58510206; file `artifacts/kaggle-t039/t039/submission-depth12_node-s42.csv` (sha256 `28b08ccf5bebddf1`).
- Public order across train-only models (S018 < S019 < S005 < S020) matches holdout order.
