# S002 - Approved T006 vector-attention checkpoint submission

The user explicitly approved one submission on November 12, 2026.
Status: preparing inference; no submission ID or public score yet.

## Exact model

Use the **unchanged best epoch-1 checkpoint from T006**, trained on 1,230,110
chronological training rows, with its original train-fitted preprocessing.
Its validation ROC AUC is 0.8270238904 on 307,528 held-out rows. This is not a
refit on all labeled rows: preserving the exact approved model is intentional.

Training kernel: `dheerajgattupalli/sharechat-custom-vector-attention-gpu`,
version 1. Training source revision:
`899b34b05c4df0130af94f7ea7fb091dfdd81d90`.

## Inference and verification

The private inference notebook attaches the original T006 outputs and competition
data. It checks the original checkpoint, preprocessing and training-result hashes,
and requires the same package versions before deserializing preprocessing.
The Kaggle image digest is pinned to the training image.

Before predicting test rows, replay the complete chronological validation cohort
through raw-data inference and require its AUC to match the recorded checkpoint
within 0.000005. Preprocessing only calls fitted `transform` methods: it does not
recompute vocabularies, medians, scales, rarity thresholds, or network weights.

Then predict all 1,037,090 test rows in bounded batches. Verify submission IDs
against the sample template, finite probabilities in [0, 1], row count, and CSV
SHA-256 before making exactly one approved competition submission.

The notebook itself never calls the competition submission API. The approved
submission is a separate CLI action after verified prediction export.

## Archive

`notebook.ipynb` freezes inference and model definitions. `provenance.json`
records source identity and expected training-artifact checksums. Original
inference results, log, executed metadata, and the grading record will be saved
here after completion. Large artifacts and the prediction CSV remain under
ignored `artifacts/kaggle-s002/`.

The inference notebook is
`dheerajgattupalli/sharechat-vector-attention-inference`. No retraining or
additional model runs are performed. Approval covers this submission only;
future submissions still require explicit user approval.
