# ShareChat RecSys Challenge 2026 — Solution Writeup

**Team:** `dheerajgattupalli`  
**Prize-Eligible Final Submission:** **S025** — Kaggle Submission ID **`58816244`** (`submission-full-rowonly-superblend.csv`)  
**Public ROC AUC:** **`0.83562`** | **Primary Chronological Validation AUC:** **`0.842412`** | **Earlier Chronological Validation AUC:** **`0.839000`**

> **Disclosure on Selected Submissions:** Of our two selected submissions on Kaggle, **`58816244` (S025, `0.83562`)** is our **strictly row-independent, prize-eligible solution** documented below. Our other selected submission (`58479943`, S017, `0.88648`) was an earlier experiment that used `request_id` grouping features; we are **not** submitting it for prize review and all code, features, and model artifacts shared here correspond strictly to **S025 (`58816244`)**.

---

## 1. Rule Compliance & Train vs. Test Feature Disclosure

1. **Zero `request_id`-based or cross-row features:** Every model in S025 predicts each evaluation row independently using **only the features of that single row** plus frozen parameters learned strictly on `train.parquet`. `request_id` is **never** read by any feature extractor, model input, hash, or encoder (it was used solely for chronological split bookkeeping and paired cluster-bootstrap evaluation intervals during validation).
2. **Identical features for train and test:** Our approach uses the **exact same features** for both training and test. There are **no** train-only features, no test-set statistics, no transductive normalization, and no test-batch rank averaging. All probabilities are combined **pointwise** per row.
3. **Row-independence verification:** Every model and feature pipeline was verified with automated single-row vs. batched and reversed-batch invariance assertions (`pred(batch)[:k] == concat(pred(row_i))`).

---

## 2. Approach Overview

Our solution is a **pointwise, multi-family ensemble** combining **19 row-independent models** across four distinct model families trained on all **1,537,638 labeled rows**:

- **Periodic-Embedding Neural Networks (`TabM k32`, `TabM k8`, `Quantile-PLR TabM k32`)** — capturing smooth nonlinear numeric patterns and 91 pairwise embedding interactions.
- **Deep & Cross Networks (`DCNv2-r32`, `DCNv2-r64`)** — explicit low-rank multiplicative feature crosses over PLR numeric embeddings, vector crosses, and low-cardinality categorical embeddings.
- **Residual Gated MLPs (`Gated-MLP` with robust scaling and train-only quantile encoding)** — lightweight, low-correlation numeric diversity.
- **Gradient Boosted Decision Trees (`XGBoost` depths 6, 8, and 12 + `LightGBM`)** — spanning four complementary row-local feature views (raw + cosines, rich embeddings + dot/bias, duration-relative interactions, paired embedding rotations, match landmarks, and super-enriched user×item rate/cosine crosses).

Progression of our strictly row-independent public submissions:
- **S003** (Single TabM k32): `0.840454` val → **`0.83349`** public
- **S004** (4-model probability blend: TabM k32/k8 + DCNv2 + LightGBM): `0.840806` val → **`0.83365`** public
- **S022** (Full-data S004 + Depth-12 Cosine XGBoost): `0.841540` val → **`0.83466`** public (`+0.00101`)
- **S023** (Full-data S004 + 4-tree Depth 6/12 Enriched XGBoost): `0.841990` val → **`0.83504`** public (`+0.00038`)
- **S024** (S023 + Duration tree + Compact tree + Paired-rotation tree + Landmark trees + Gated/Quantile MLPs + hybrid prob/logit pooling): `0.842296` val → **`0.83551`** public (`+0.00047`)
- **S025** (7/11 S024 + 2/11 Super-Enriched `xgb_d6` + 1/11 2-seed Quantile-PLR `TabM k32` + 1/11 Super-Enriched `DCNv2-r64`, logit blend): **`0.842412`** val → **`0.83562`** public (`+0.00011`)

---

## 3. Complete List of Every Feature Used for Evaluation Predictions

Every feature below is computed strictly from the current row of `test.parquet` (using frozen training-only parameters where noted). **`request_id` is excluded everywhere.**

### A. Raw Row Features
1. **584 Raw Numeric Scalars:**
   - `feature3` (hour of day, Int8)
   - All 583 scalar numeric columns (`Float32`, `Int64`, `Int8`) among `feature*` excluding `feature1` (raw datetime) and `feature3`.
   - *Note:* For neural models, missing values are imputed with train-only medians (with binary missingness indicators in T021) and scaled using train-only mean/std, robust IQR, or 33-knot train-only quantile percentiles.
2. **14 Raw 32-Dimensional Embedding Lists (`448` coordinates):**
   - 7 user-side embeddings: `feature26` through `feature32` (32 floats each)
   - 7 post/row-side embeddings: `feature40` through `feature46` (32 floats each)
   - Missing lists are zero-filled with an explicit binary `present` flag (`norm > 1e-6`).
3. **Low-Cardinality Categoricals (train-only vocabulary, high-cardinality masked):**
   - Used in neural models (`T021`, `T096` `DCNv2-r64` and `q_tabm`) and `T021` LightGBM:
   - Candidate fields: `feature6, feature7, feature8, feature9, feature10, feature11, feature12, feature13, feature14, feature16, feature17, feature19, feature21, feature22, feature25` (plus `feature2, feature4, feature22` in T021).
   - **High-cardinality masking:** Any categorical column with `> 1,000` unique values in `train.parquet` (including `feature5`, `feature18`, `feature20`, `feature23`, `feature24`) is masked to index `0` in neural models to prevent ID memorization. Retained values must occur `≥ 10` times in `train.parquet`; unseen/missing test values map to index `0`.

### B. Engineered Row-Local Features (All Computed Independently Per Row)
1. **Embedding Geometry & Latent-Score Interactions:**
   - **14 L2 Norms** (`||v_k||_2` and `log1p(||v_k||_2)`) and **14 presence indicators** for `feature26–32` and `feature40–46`.
   - **49 User × Post Cosines:** cosine similarity between every pair `(u, v)` for `u ∈ {feature26..32}` and `v ∈ {feature40..46}` (T062/T066 use the 10 strongest pairs: 7 diagonal + `(32,43), (30,46), (31,44)`; T096 uses all 49).
   - **49 Dot + Adjacent-Bias Latent Scores:** `dot(u, v) + b_u + b_v` for every `(u, v)` pair, where `b_u ∈ {feature33..39}` and `b_v ∈ {feature47..53}`.
   - **364 Pairwise Vector Crosses (inside neural models T021 & T096):** for all `C(14, 2) = 91` pairs among the 14 embeddings: `[cosine * present, sign(dot) * log1p(|dot|) * present, log1p(L2_distance) * present, present]`.
2. **36 Video-Duration Relative Features (`T069 / T087 / T096`):**
   - For video duration `d = feature54` (when `d > 0`, else `NaN`) and 18 watch-time history scalars `s ∈ {feature476, 477, 479, 480, 483, 484, 486, 487, 492, 493, 500, 501, 502, 503, 512, 513, 514, 515}`:
     - 18 log-ratios: `log1p(s) − log1p(d)`
     - 18 signed differences: `s − d`
3. **60 Train-Fitted Paired Embedding Rotation Features (`T083 / T087`):**
   - Regularized whitening + SVD cross-covariance alignment fitted strictly on a 32,768-row training sample for the 10 user×post vector pairs (rank 12 per pair). At test time, each row’s embeddings are projected via the frozen linear matrices to produce 4 leading coordinate products, mean product, and aligned cosine per pair (`10 × 6 = 60` features).
4. **72 Train-Fitted Match Landmark Distances (`T086 / T087`):**
   - For 3 core embedding pairs `(feature32, feature43)`, `(feature30, feature46)`, `(feature31, feature44)`, 24 representative points in `[unit(u), unit(v), unit(u)*unit(v)]` space (one model using `MiniBatchKMeans` centers, one using random training exemplars — both fitted strictly on a 32,768-row training sample). Test rows compute 72 squared Euclidean distances to these frozen points.
5. **60 Super-Enriched Domain Interactions (`T092 / T094 / T095 / T096`):**
   - **5 Temporal Features (from `feature1` and `feature57/58/59`):** `log1p(age_days)` for `feature57` (ms/s normalized), `feature58` (s), and `feature59` (ms), plus `sin(2π·hour_frac)` and `cos(2π·hour_frac)`.
   - **32 User × Post Watch-Rate Interactions:** for `u ∈ {feature352, feature346, feature358, feature334}` and `v ∈ {feature418, feature419, feature430, feature411}`: both `u * v` (16 features) and `logit(u) + logit(v)` (16 features).
   - **11 Top-Cosine Crosses:** `cos(32,43)` and `cos(30,46)` each multiplied by `{logit(feature418), logit(feature352), log1p(feature54), log1p(feature493)}`, plus `cos(32,43) + cos(30,46)`, `cos(32,43) − cos(30,46)`, and `cos(32,43) * cos(30,46)`.
   - **9 User vs. Post Watch-Time Log-Ratios:** `log1p(u) − log1p(v)` for `u ∈ {feature493, 480, 484}` and `v ∈ {feature515, 511, 502}`.
   - **3 Watch-to-Skip Ratios:** `feature352 / (feature358 + 1e-4)`, `feature346 / (feature334 + 1e-4)`, and `feature418 / (feature430 + 1e-4)`.

---

## 4. Model Architectures & Exact Ensemble Composition (S025)

All 19 models in **S025** were refitted on all **1,537,638 labeled rows** using validation-frozen training budgets (no test labels or early stopping on test):

### Stage 1: Base Row-Only Anchor (`T021` — 4 models, fixed probability average)
| Model | Architecture / Key Settings | Full-Data Budget | Weight in T021 |
|---|---|---:|---:|
| `tabm32` | PLR (`PeriodicEmbeddings`, d=8, 8 freq) + 91×4 vector crosses + masked low-card cats + `TabM` (`k=32`, 2 blocks, width 128, dropout 0.2, AdamW lr=5e-4) | 11,621 steps (7.74 ep) | 45.0% |
| `tabm8` | Same as `tabm32` with `k=8` | 11,621 steps (7.74 ep) | 22.5% |
| `dcn32` | Same encoder + 2-layer Low-Rank `DCNv2` (rank 32, deep MLP 128-128) | 2,624 steps (1.75 ep) | 22.5% |
| `lightgbm` | 31 leaves, lr=0.03, `feature_fraction=0.8`, `min_data_in_leaf=100`, L2=1 | 140 trees | 10.0% |

### Stage 2: S024 Landmark Blend (`T062 + T066 + T087` — 11 additional models)
- **4 Core XGBoost models** (`T062/T066`, `eta=0.02`, 900 rounds, `max_bin=127`): `d12-r0` (depth 12, 594 feats), `d12-r1` (depth 12, 1,105 feats), `d6-r0` (depth 6, 594 feats), `d6-r1` (depth 6, 1,105 feats).
- **Duration XGBoost** (`T087`, depth 6, `eta=0.02`, 2,400 rounds, 1,141 feats).
- **Compact XGBoost** (`T087`, depth 8, `eta=0.025`, 1,600 rounds, 679 feats: 594 + 36 duration + 49 dot/bias).
- **Paired-Rotation XGBoost** (`T087`, depth 8, 1,600 rounds, 679 + 60 SVD-aligned features).
- **2 Match-Landmark XGBoost models** (`T087`, depth 8, 1,600 rounds, 679 + 72 KMeans / random landmark distances).
- **2 Gated Residual MLPs** (`T087`, 192-wide 2-block residual + multiplicative gate, 8 epochs, 1,141 feats): one with robust IQR scaling, one with 33-knot train-only quantile encoding.
- **S024 Pointwise Pooling Formula:**
  - Let `q = [T021, d12-r0, d12-r1, d6-r0, d6-r1, duration, gated_mlp, compact]` with weights `w = [0.346215625, 0.08655390625×4, 0.09891875, 0.04165, 0.167]`.
  - Hybrid pool: `b = 0.25 * (q @ w) + 0.75 * sigmoid(logit(q) @ w)`.
  - `S024 = 0.95 * (0.90 * (0.95 * b + 0.05 * quantile_mlp) + 0.10 * paired_xgb) + 0.025 * (learned_landmark_xgb + random_landmark_xgb)`.

### Stage 3: S025 Super-Enriched Upgrade (`T096` — 4 additional models on 1,240 features)
| Model | Architecture / Key Settings | Full-Data Budget | Weight in S025 Logit Blend |
|---|---|---:|---:|
| **S024** | 15-model row-only ensemble above | Frozen | **7 / 11 (63.64%)** |
| **`t096_xgb_d6`** | GPU XGBoost depth 6, `eta=0.02`, `colsample_bytree=0.4`, `colsample_bynode=0.5`, `min_child_weight=150`, `reg_lambda=20`, `reg_alpha=2`, 1,240 super-enriched features | 1,125 rounds | **2 / 11 (18.18%)** |
| **`t096_q_tabm` (2 seeds)** | 2-seed average (`seeds 42, 43`) of 33-knot train-only **Quantile-PLR `TabM k32`** (width 256, 2 blocks, 792 dense + 476 vec + 364 crosses + low-card cats, CosineAnnealingLR `4.5e-4 → 5e-5`) | 8 epochs each | **1 / 11 (9.09%)** |
| **`t096_dcnv2_r64`** | 3-layer Low-Rank **`DCNv2` (rank 64, width 256)** + PLR + 364 vector crosses + low-card cats (CosineAnnealingLR `4e-4 → 5e-5`) | 4 epochs | **1 / 11 (9.09%)** |

Final pointwise prediction:
```python
p_S025 = sigmoid((7 * logit(p_S024) + 2 * logit(p_xgb_d6) + 1 * logit(0.5 * (p_qtabm_s42 + p_qtabm_s43)) + 1 * logit(p_dcnv2_r64)) / 11.0)
```

---

## 5. Training & Validation Protocol

We enforced strict **chronological out-of-time validation** on two predeclared temporal partitions (no random row splits; timestamp ties and requests never cross a split boundary):

| Partition | Training Rows | Validation Rows | Cutoff / Window |
|---|---:|---:|---|
| **Primary Holdout** | 1,230,110 | 307,528 | `< 2026-10-28 08:06:52.397` / final 10 hours |
| **Earlier Holdout** | 710,330 | 299,969 | `< 2026-10-27 08:06:52.397` / next ~10 hours to `18:00` |

- **Anti-overfitting discipline:** Blend weights and candidate selections were chosen on the **earlier** chronological fold and evaluated **frozen** on the **primary** holdout before promoting any recipe to full-data refit.
- **Confirmed Two-Fold Progression:**
  - `T016/S004` anchor: Primary `0.840806` | Earlier `0.837549` (T018 proxy) → Public `0.83365`
  - `S023` (4-tree blend): Primary `0.841990` | Earlier `0.838514` → Public `0.83504`
  - `S024` (landmark blend): Primary `0.842296` | Earlier `0.838857` → Public `0.83551`
  - **`S025` (final super-blend):** Primary **`0.842412`** (`+0.000115` vs S024) | Earlier **`0.839000`** (`+0.000143` vs S024) → Public **`0.83562`** (`+0.00011` vs S024)
- **Compute environment:** Kaggle NVIDIA T4 GPUs (Python 3.12, PyTorch, `tabm==0.0.3`, `rtdl_num_embeddings==0.0.12`, XGBoost GPU `hist`, LightGBM, Polars, PyArrow, scikit-learn) and local Apple Silicon CPU for verification and assembly.

---

## 6. Code & Trained Model Artifacts

- **Repository:** All frozen training scripts, executed Kaggle notebooks, assembly scripts, validation JSONs, and submission receipts are archived under `trials/` and `submitted/025-row-only-super-blend/` (and `submitted/022–024/`).
- **Key Reproduction Files:**
  - T021 (S004 full refit): `trials/021-full-data-finalist-refit/notebook.ipynb` (`refit_finalists.py`)
  - T062 / T066 / T087 / T096 (15 full-data models + trained `.ubj`/`.pt`/preprocessor export): `trials/098-s025-artifact-export/notebook.ipynb` and `trials/096-full-row-only-super-blend/assemble.py`
  - Final submission SHA-256 (`submission-full-rowonly-superblend.csv`, 1,037,090 rows): `e81c9b772a8f409ebf47767787dd4c7a8f2cbf344bcff78bcec9520990eb6e8f`
