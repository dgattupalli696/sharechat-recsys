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
