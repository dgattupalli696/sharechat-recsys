# T021 - Full-data refit of the selected finalist recipe

The user approved full-data training after held-out model/approach selection.
T019 is complete and T016 remains the selected ensemble. This run trains final
models and generates test predictions only; it is not another validation
experiment and does not upload a competition submission.

## Frozen selection

| Component | Fixed training budget | Final probability weight |
| --- | ---: | ---: |
| PLR/cross/masked TabM k32, seed 42 | 7.7371048253 epochs | 45% |
| PLR/cross/masked TabM k8, seed 42 | 7.7371048253 epochs | 22.5% |
| PLR/cross/masked DCNv2-r32, seed 42 | 1.7470881864 epochs | 22.5% |
| Expanded-feature LightGBM, seed 42 | 140 trees | 10% |

Neural duration is frozen from each original T014 selected checkpoint as a
fraction of complete epochs; scale optimizer steps to the larger population,
rounding to the nearest step. Keep batch 1,024, AdamW learning rate 0.0005,
weight decay 0.01 and the original architectures. No early stopping, checkpoint
search, ensemble-weight tuning, correlation-based feature changes or new
graph/history features.

Train on **all 1,537,638 labeled rows**, including the previous holdout. Refit
medians, missingness indicators, standardization, categorical vocabularies and
rare-category mappings on that full training population, never on test rows.
Retain the original masking rule: minimum count 10; mask fields with original
cardinality above 1,000. Recompute the same numeric-embedding eligibility rule
on full-training features. The initial prior is the full-training positive rate.

## Bounded-memory implementation

Stream scalar/vector summaries into a temporary numeric memmap. Fit the original
median imputer on all rows, then update the same standard scaler in blocks.
Learn categorical support and counts from complete training columns; use the
public fixed-category OrdinalEncoder API to initialize that exact vocabulary.
Delete the summary cache before streaming final dense/vector features.

Neural batches index the chronologically ordered complete training population.
Every component visits all rows in its first full epoch. Save the fixed endpoint
and check exact checkpoint reload probabilities. Then stream test data through
all three saved neural checkpoints.

Fit/predict LightGBM in separate processes to avoid native OpenMP conflicts with
Torch. Use disk-backed training features and delete them before generating the
test matrix. Preserve original feature order and train-fitted categorical codes.

Export both the full-data TabM k32 and selected ensemble CSVs, with exactly
1,037,090 sample-template-ordered IDs and finite probabilities in [0,1].
Archive all component probabilities, hashes, full-training counts, source,
image and package versions. Report no holdout AUC for these refitted models;
0.8408057533 is selection evidence from T016, not a new evaluation result.

**No new leaderboard submission is approved.**

## Completed

Canonical Kaggle notebook version 14 completed all four refits in 2,853 seconds.
Both TabM models used 11,621 optimizer steps and DCNv2 used 2,624; LightGBM
used exactly 140 trees. All 1,537,638 labels were included.

Independent verification confirmed frozen source/image, checkpoint hashes,
full-training preprocessing/label counts, all 1,037,090 ordered test IDs and
exact fixed-weight recombination. Both CSVs are ready under
`artifacts/kaggle-t021/refit/`: `submission-full-tabm.csv` and
`submission-full-ensemble.csv`. Neither was uploaded to the competition.

See `results.json` and `independent-verification.json`. The separately requested
T022 checkpoint-only former-holdout check is explicitly in-sample, not a new
validation result.
