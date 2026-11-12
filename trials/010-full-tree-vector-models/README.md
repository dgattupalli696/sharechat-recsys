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

PCA uses all present training vectors, never validation/test rows, preserving
summary and presence features. Native embedding support uses the explicit
LDA-only calcer configuration rather than a nearest-neighbor pipeline.
Anonymous vector cross-features are experimental, not claims about semantics.

LightGBM moved to [T011](../011-local-full-lightgbm/README.md), running on
the local CPU in parallel with Kaggle. This unlaunched batch now explicitly
selects only the five CatBoost variants; no duplicate LightGBM job is launched.
The frozen model implementations are unchanged. Scientific stack versions
and actual tree-library versions/backends are recorded.

Reuse the existing versioned Kaggle notebook. Archive original metrics, logs,
source and configuration identity; keep large artifacts outside Git.
No test predictions or competition submission.

## Completed full-data results

Kaggle notebook version 5 completed all five CatBoost variants.
All predictions match the full primary holdout's labels and independently
reproduce the reported AUCs.

| Variant | Validation AUC |
| --- | ---: |
| catboost-expanded | 0.8164893363 |
| pca-8 | 0.8162657347 |
| pca-16 | 0.8160712030 |
| catboost-crosses | **0.8252509472** |
| catboost-native-lda | 0.8167170746 |

Expanded CatBoost exactly reproduces T003. Exploratory vector crosses help
materially, while PCA and native LDA embeddings do not beat that cross-feature
recipe. T011 LightGBM remains the strongest individual tree at 0.8255838643.
Preserve all predictions for later ensemble comparison; do not infer that
the weaker individual models cannot add ensemble diversity.
