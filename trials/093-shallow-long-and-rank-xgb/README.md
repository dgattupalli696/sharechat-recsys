# T093 – Shallow long-horizon XGBoost (d4 stationary, d5 long) and global pairwise-rank XGBoost

Follow-up to T092 (where `xgb_d5` was still rising at round 2,000 and hit 0.841488 standalone). Train three distinct XGBoost recipes on the 1,240 super-enriched row-only features across both chronological folds:
1. `xgb_d4_stat`: depth 4, lr 0.015, heavy regularization (`reg_lambda=50, reg_alpha=8, colsample_bytree=0.25`), 2,400–4,200 rounds, **dropping raw timestamps `feature57/58/59`** so only stationary log-ages remain.
2. `xgb_d5_long`: depth 5, lr 0.015, `reg_lambda=40, reg_alpha=6, colsample_bytree=0.30`, 2,000–3,600 rounds.
3. `xgb_d5_rank`: depth 5, **`objective='rank:pairwise'`** (global AUC optimization, zero `request_id`), 1,200–2,400 rounds.
Strictly row-independent; singleton invariance verified. No submission.
