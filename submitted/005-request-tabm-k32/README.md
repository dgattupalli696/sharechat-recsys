# S005 — Request-context TabM k32

User approval: "can we try submissiong with tabm k32 and ensemble?"

- Model: unchanged T024 `ensemble-32` checkpoint (TabM k32 + 33 label-free request-context features); no refit.
- Original holdout (307,528 rows) AUC: 0.889350 (replayed on Kaggle, max probability error 7.5e-8).
- Inference: private kernel `sharechat-model-experiments-gpu` v22, T4; bundle `sharechat-s005-s006-checkpoints`.
- Submission 58300770: **public 0.88394** (S003: 0.83349).

Files: `notebook.ipynb`, `executed-notebook.ipynb`, `run.log`, `inference-results.json`, `independent-verification.json`, `submission-receipt.json`, `leaderboard-submission.json`.
