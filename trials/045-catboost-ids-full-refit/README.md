# T045 - CatBoost + raw IDs, full-data refit

Refits the T041 `catboost_ids` model (holdout 0.88829, best iteration 2339) on all
1,537,638 labeled rows for 2925 iterations (best iteration + 1, scaled by
1537638/1230110), seeds 42/43/44, with `xgb_tune.py catboost_refit`. It is the only
member of the 5-model blend (T027 + T024 k32 + xgb511 + xgbd12 + CatBoost-IDs,
holdout 0.89102) without a full-data refit. Kernel: `sharechat-model-experiments-gpu`.
