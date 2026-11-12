# S002 - Approved T006 vector-attention checkpoint submission

**Public leaderboard ROC AUC: 0.81905.**

The user explicitly approved this one submission on November 12, 2026.

| Field | Value |
| --- | --- |
| Submission ID | 58275874 |
| Submission time | November 12, 2026, 14:29:42.673 UTC |
| Status | Complete |
| Public score | 0.81905 |
| Previous baseline S001 | 0.80699 |
| Change versus S001 | +0.01206 |
| Test prediction rows | 1,037,090 |
| Refitting | None; original T006 epoch-1 checkpoint |

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

Validation replay produced **0.8270238905 AUC**, versus the original
**0.8270238904**; the difference is approximately 0.0000000001. Original checkpoint
and preprocessing hashes matched exactly. Inference and replay together took
256.35 seconds, excluding kernel startup. The CSV was verified again locally
before the single upload.

## Archive

- `notebook.ipynb`: frozen checkpoint-only inference and model definitions.
- `provenance.json`: training/inference source identity, original model checksums,
  exact CSV checksum, kernel versions, and submission reference.
- `inference-results.json`: original replay, probability-export, and environment record.
- `run.log`: original inference execution log.
- `executed-kernel-metadata.json`: executed notebook configuration and image digest.
- `submission-receipt.json`: successful single-upload receipt.
- `leaderboard-submission.json`: original Kaggle grading record.

Large artifacts and the prediction CSV remain under ignored
`artifacts/kaggle-s002/`.

The inference notebook is
`dheerajgattupalli/sharechat-vector-attention-inference`, version 1.
Inference source revision: `19e849b0a3ad0c6129917d4c0ec4f8b52e5f228f`.

No retraining or additional model runs were performed. Approval covered this
submission only and has been consumed; future submissions still require
explicit user approval.
