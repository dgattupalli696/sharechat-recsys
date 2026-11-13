# S004 - Approved highest validation ensemble

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

Upload this ensemble CSV once after verified test IDs, probability bounds,
row count and exact file checksum. Approval covers only this and S003.
