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
