# T018 - Full-data seed and temporal confirmation

Run the original fine-checkpoint TabM reference and two frozen finalists
with seeds 42, 43 and 44 on both complete predefined temporal partitions:
18 full fits, without subset screens. Do not select new input recipes,
search new weights, predict the competition test set or submit.

| Model | Primary seed-42 selection result | Reason |
| --- | ---: | --- |
| Original reference | 0.8292340756 | Matched historical anchor |
| PLR/cross/masked TabM k32 | 0.8404542697 | Best individual full-data model |
| PLR/cross/masked DCNv2-r32 | 0.8389150778 | Strong complementary finalist: its 25% vote with k32 gave 0.8406937050 |

These are the original configurations from T008/T014, not reduced models.
The DCNv2 finalist is selected for observed complementary errors, not for
being the second-highest standalone AUC. The k8 member variant is not
retrained here; do not describe this experiment as a complete seed/time
confirmation of the four-component T016 ensemble.

| Partition | Train | Validation | Cutoff / exclusive validation end |
| --- | ---: | ---: | --- |
| Primary | 1,230,110 | 307,528 | October 28, 2026, 08:06:52.397 / dataset end |
| Predeclared earlier | 710,330 | 299,969 | October 27, 2026, 08:06:52.397 / October 27, 2026, 18:00 |

Each fold uses all available prefix training rows, train-only preprocessing,
the complete validation population and original row identities. Share
preprocessing only among recipes/seeds on the same fold. Run the two folds
in separate fresh processes to release allocator memory between them.
Keep validation every 300 optimizer updates and at epoch boundaries,
five epoch-equivalent patience and a 30-epoch maximum.

Repeat primary seed 42 rather than silently reusing scores, providing an
exact-configuration reproducibility check. Report per-seed AUC/logloss,
AUC mean/sample standard deviation, resource costs and paired
request-clustered uncertainty. Use the same frozen weights, if comparing
the preselected 75% k32 / 25% DCN vote across folds; never tune them on the
earlier fold. This is robustness confirmation, not a retroactively untouched
test set: model/weight selection already used the primary holdout, and
early stopping still uses each fold's validation labels.

T017's full two-base stack scored 0.8394886338, below T016's 0.8408057533
best blend; it does not replace these finalists. The GNN follow-up remains
deferred. No competition submission.

## Completed full-data results

All 18 fits completed on canonical notebook v9. The original primary
seed-42 probabilities, targets and identities reproduced exactly for all
three recipes. Values below are three-seed AUC mean +/- sample standard
deviation, not confidence intervals.

| Model | Primary | Earlier |
| --- | ---: | ---: |
| Reference | 0.829399 +/- 0.000143 | 0.824690 +/- 0.000589 |
| TabM k32 | **0.840305 +/- 0.000156** | **0.837033 +/- 0.000051** |
| DCNv2-r32 | 0.838849 +/- 0.000192 | 0.835442 +/- 0.000067 |

| Model | Primary mean logloss | Earlier mean logloss |
| --- | ---: | ---: |
| Reference | 0.1586804 | 0.1664664 |
| TabM k32 | **0.1553804** | **0.1631073** |
| DCNv2-r32 | 0.1562876 | 0.1637575 |

Total recorded model fitting/prediction time was 2.914 GPU hours, plus
8.88 minutes preparing the two full folds. The individual per-seed AUC,
logloss, selected checkpoint and duration are in `results.json`.

## Fixed seed averaging and frozen vote

Average probabilities across the same three seeds, then apply the already
selected 75% k32 / 25% DCN mixture. No new weight search on either fold.

| Seed-averaged prediction | Primary AUC | Earlier AUC |
| --- | ---: | ---: |
| Reference | 0.8305613051 | 0.8258823541 |
| TabM k32 | 0.8406124465 | 0.8374046118 |
| DCNv2-r32 | 0.8400306532 | 0.8366054339 |
| Frozen 75/25 vote | **0.8407525120** | **0.8375490212** |

The frozen vote improves the k32 seed mean by 0.0001400655 on primary
and 0.0001444094 on earlier. Conditional 500-replicate request-bootstrap
95% delta intervals are [+0.0000357, +0.0002497] and
[+0.0000391, +0.0002451], respectively. These intervals are neither
selection-adjusted nor time-block-adjusted.

T016 remains the best observed primary holdout score, 0.8408057533.
The frozen six-model vote differs by -0.0000532414, with conditional
95% delta interval [-0.0002475, +0.0001236]; the difference is inconclusive.
This confirms the improved model family and complementary errors across
both periods, not the exact four-component T016 mixture. Preserve both
candidates; do not claim independent test/leaderboard confirmation.

`finalists.json` freezes the selected k32 recipe, the best observed holdout
mixture and the two-fold-confirmed vote separately. `analysis-results.json`
records source fingerprints and bootstrap results.
From the repository root, `python trials/018-full-seed-time-confirmation/analyze_results.py`
recreates the fixed postprocessing from the saved model predictions.
