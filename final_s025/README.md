# Final Prize-Eligible Submission — S025 (`58816244`)

**Team:** `dheerajgattupalli`  
**Kaggle Submission ID:** `58816244` (`submission-full-rowonly-superblend.csv`)  
**Public ROC AUC:** **`0.83562`** | **Primary Chronological Validation AUC:** **`0.842412`** | **Earlier Validation AUC:** **`0.839000`**  
**Serving Compliance:** 100% strictly row-independent. Zero `request_id`-derived features, zero cross-row features, identical features for train and test, pointwise inference and blending.

## Contents of this Folder

- [`SOLUTION_WRITEUP.md`](SOLUTION_WRITEUP.md) — Complete solution writeup, including an exhaustive list of every feature used for evaluation predictions, train vs. test feature disclosure, architectures, ensemble weights, and chronological validation results.
- [`EMAIL_DRAFT_SHARECHAT.md`](EMAIL_DRAFT_SHARECHAT.md) — Ready-to-send email reply template for `saira@sharechat.co`.
- `code/` — Frozen, self-contained code and metadata for S025:
  - `t021_notebook.ipynb` — Full-data refit of the 4 base anchor models (`tabm32`, `tabm8`, `dcn32`, `lightgbm`).
  - `t098_export_notebook.ipynb` & `t098_worker.py` — Full-data refit and artifact export of the 15 additional models (`T062`, `T066`, `T087`, `T096`) and all frozen preprocessors.
  - `assemble_s025.py` — Exact pointwise logit-blend assembly script (`7/11 S024 + 2/11 xgb_d6 + 1/11 q_tabm_2s + 1/11 dcnv2_r64`).
  - `submission-receipt.json`, `candidate.json`, `validation.json` — Official Kaggle submission receipt, SHA-256 checksum (`e81c9b772a8f409ebf47767787dd4c7a8f2cbf344bcff78bcec9520990eb6e8f`), and two-fold validation metrics.

## Trained Model Artifacts (All 19 Models + Preprocessors)

All 19 trained model files (`.ubj`, `.pt`, `.txt`) and every frozen train-only preprocessor (`.joblib`, `.npz`) are attached to GitHub Release **`s025-final`**:

- **Direct download:** `https://github.com/dgattupalli696/sharechat-recsys/releases/tag/s025-final` (`s025-trained-model-artifacts.zip`, 96.9 MiB, includes SHA-256 `MANIFEST.json`).
