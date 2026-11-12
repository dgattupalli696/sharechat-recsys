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

## Completed full-holdout results

Evaluated 2,808 configurations from thirteen distinct candidates.

| CatBoost baseline weight in the frozen T015 ensemble | Validation AUC |
| --- | ---: |
| 0% | 0.8407075087 |
| 1% | 0.8407204822 |
| 2.5% | **0.8407288153** |
| 5% | 0.8407100894 |
| 10% | 0.8405723544 |
| 25% | 0.8394669302 |
| 50% | 0.8353575230 |

The original baseline contributes only a tiny holdout-selected gain.
The best overall mix instead adds **10% LightGBM** to 90% of the frozen
T015 ensemble, giving **0.8408057533 AUC**. Expanded into base-model weights:
45% TabM k32, 22.5% TabM k8, 22.5% DCNv2-r32 and 10% LightGBM.

Its gain over T015 is 0.0000982446. The 500-replicate conditional
request-bootstrap 95% delta interval is [-0.0000197, +0.0002286], including
zero even before adjusting for weight/model selection. Do not treat this
small difference as established or replace temporal confirmation with it.
