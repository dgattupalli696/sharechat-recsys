# S025 — full-data row-only super-blend

Kaggle submission **`58816244`** completed Nov 24, 2026: **0.83562 public ROC AUC** (+0.00011 vs S024).

User approval: "Lets make a submission with our best so far?"
T096 Kaggle GPU v59 full refit on all 1,537,638 labeled rows, combining in logit space:
- 7/11 S024 (`artifacts/local-t087/submission-full-rowonly-landmarks.csv`)
- 2/11 T092 super-enriched `xgb_d6` (1,125 rounds, 1,240 row-only features)
- 1/11 T095 2-seed Quantile-PLR TabM k32 width-256 (`ep8`, seeds 42 & 43)
- 1/11 T094 super-enriched `dcnv2_r64` (`ep4`)

Strictly row-independent, zero `request_id` usage, singleton invariance verified.
Validation AUC: **0.842412** primary (+0.000115 vs S024), **0.839000** earlier (+0.000143 vs S024).
File: `artifacts/local-t096/submission-full-rowonly-superblend.csv`, SHA-256 `e81c9b772a8f409ebf47767787dd4c7a8f2cbf344bcff78bcec9520990eb6e8f`, 1,037,090 rows.
