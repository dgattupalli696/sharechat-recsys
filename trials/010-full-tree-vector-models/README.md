# T010 - Full-data tree and vector-feature comparisons

Train on all 1,230,110 rows before October 28, 2026, 08:06:52.397 and evaluate
on the identical 307,528-row holdout. No variant is screened on a subset.

| Variant | Configuration |
| --- | --- |
| catboost-expanded | Reproduce expanded CatBoost: GPU, depth 6, LR 0.06, maximum 400 trees, patience 60 |
| pca-8 | Replace each field's raw coordinates with eight training-fitted PCA coordinates |
| pca-16 | Same, with sixteen coordinates |
| catboost-crosses | Summary inputs plus exploratory dot/cosine/L2/presence features |
| catboost-native-lda | Native 32-D vector inputs, CPU LDA embedding calcers, same tree settings |
| lightgbm | CPU binary logloss, AUC selection, LR 0.03, 31 leaves, min child 100, max 1,000 trees, patience 60 |

PCA uses all present training vectors, never validation/test rows, preserving
summary and presence features. Native embedding support uses the explicit
LDA-only calcer configuration rather than a nearest-neighbor pipeline.
Anonymous vector cross-features are experimental, not claims about semantics.

LightGBM runs in an isolated process to avoid conflicting native OpenMP
runtimes. Temporary full-feature frames are removed after the worker finishes;
only models, probabilities and small metrics persist as outputs. Scientific
stack versions and the actual tree-library versions/backends are recorded.

Reuse the existing versioned Kaggle notebook. Archive original metrics, logs,
source and configuration identity; keep large artifacts outside Git.
No test predictions or competition submission.
