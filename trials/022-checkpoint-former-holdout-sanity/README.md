# T022 - Inference-only sanity check on the former holdout

The user explicitly requested evaluation only: no additional training.

After the existing T021 job completes, load its three full-data neural
checkpoints, full-training preprocessing/vocabularies and 140-tree LightGBM
model. Evaluate each component and the unchanged 45/22.5/22.5/10 ensemble on
the original 307,528-row primary holdout, selected by original source row IDs.

**These are in-sample sanity scores.** Every evaluated row was part of T021's
1,537,638-row training population. Do not compare these scores as if they were
the original out-of-sample validation scores, use them to select/tune a model,
or claim that a higher score demonstrates improved generalization.

Only deserialize, transform and predict. Verify all checkpoint/preprocessing
hashes and the original package environment before loading. Do not refit
preprocessing, remap categories with older holdout-only mappings, update model
weights, change ensemble weights or submit a competition prediction file.

Use the complete original row cohort and verify its original labels/order.
Run native LightGBM inference in a separate process. Save all four component
probabilities, the fixed ensemble and an explicitly labeled in-sample AUC/log
loss report. No new training or test-set prediction generation is needed.

## Completed results

Canonical Kaggle notebook version 15 completed inference only. Independent
verification confirmed all checkpoint/preprocessing hashes, frozen source and
environment, every original row ID/label, and exact ensemble recombination.
All scores were independently recomputed from saved probabilities.

| Saved full-data model | In-sample ROC AUC | In-sample log loss |
| --- | ---: | ---: |
| TabM k32 | 0.846228516 | 0.154292553 |
| TabM k8 | 0.847834334 | 0.153825396 |
| DCNv2-r32 | 0.841303632 | 0.155258363 |
| LightGBM, 140 trees | 0.851725568 | 0.152833131 |
| Unchanged four-model ensemble | 0.848532901 | 0.153254284 |

**All 307,528 evaluated rows overlap full-data training.** LightGBM's higher
in-sample AUC does not justify replacing the holdout-selected ensemble or
retuning weights. T016 remains the independent model-selection evidence.
No training, preprocessing fitting, model selection or competition submission
was performed by this evaluation.

See `results.json`, `independent-verification.json` and `run.log`. Predictions
are archived at `artifacts/kaggle-t022/sanity/former-holdout-predictions.npz`.
