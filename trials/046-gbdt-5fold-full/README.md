# T046 - 5-fold GBDT on all labeled rows

`xgb_tune.py kfold`: GroupKFold(5) by request over all 1,537,638 labeled rows, in-fold
early stopping (300), seeds 42+fold. Configs: XGBoost depth12_node and leaves511_eta02
(T036) and CatBoost + raw IDs (T041). Outputs OOF, per-fold and fold-mean test
predictions, to be averaged with the full refits (T038, T045). A local LightGBM check
found 5-fold mean 0.88627 vs 3-seed full fit 0.88613; their rank mean 0.88641.
Kernel: `sharechat-full-refit-xgb`.
