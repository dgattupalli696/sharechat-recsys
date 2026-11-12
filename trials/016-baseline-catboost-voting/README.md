# T016 - Original CatBoost baseline in soft voting

Replay the original B001 **validation** model, not the model refitted on all
labeled rows for submission. `artifacts/kaggle-v2/validation.cbm` has 21 trees
and 679 summary features; CPU replay independently reproduces
**0.8160260803226133 AUC** on every one of the 307,528 original held-out rows.
Save original row IDs, labels, model hash and aligned probabilities.
No retraining or test inference.

Add these predictions and the frozen T015 winner to T015's eleven-model pool.
Compare soft probability voting, not thresholded class votes. Include all
single-model endpoints and the existing winning ensemble, so adding a weak
baseline is not mandatory.

Run equal means, pair/triple mixtures, and positive four-family grids with
CatBoost replacing LightGBM as the tree. Pair weights additionally include
1%, 2.5%, 5%, 10% and 12.5% contributions (and their complements), allowing a
small baseline weight without forcing 25% or more into the winner.

All comparisons use the same complete primary holdout. Weights/membership
are holdout-selected; conditional cluster bootstrap intervals are not
independent confirmation. Large outputs stay in `artifacts/local-t016/`.
No competition submission.
