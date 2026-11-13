# S004 - Submitted highest validation ensemble

Submission **58291773**, November 13, 2026: public AUC **0.83365**.
The single approved upload is complete; its approval is consumed.

Approved together with S003 on November 13, 2026. Preserve the exact T016
prediction recipe, validation AUC 0.8408057533; no retraining or weight
search.

| Saved seed-42 model | Probability weight |
| --- | ---: |
| PLR/cross/masked TabM k32 | 45% |
| PLR/cross/masked TabM k8 | 22.5% |
| PLR/cross/masked DCNv2-r32 | 22.5% |
| Expanded-feature LightGBM, 140 trees | 10% |

Use the shared S003 private inference notebook and full validation replay.
The original CatBoost baseline is not in this winning ensemble.
LightGBM prediction runs in an isolated, bounded-memory process; never
load its OpenMP runtime into the Torch process.

Shared inference completed in canonical Kaggle notebook version 10. Independent
full-holdout replay reproduced AUC 0.8408057533, with maximum probability error
3.35e-8. All 1,037,090 test rows match the sample-template ID order.

The exact shared executed notebook, environment and independent replay are in
`../003-tabm-k32/`; this folder records the ensemble CSV checksum and scored
submission receipt. Public AUC is 0.00016 above S003 and 0.01460 above S002.
Approval covered only this and S003; exactly two new uploads were made.
