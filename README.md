# ShareChat RecSys Challenge

Predict the probability that a user watches a video. Competition metric: ROC AUC.

## Run ledger

The campaign began November 12 and continues through November 25, 2026.
A trial is not a leaderboard
submission unless it has an entry in `submitted/`.

| ID | Run | Compute / labeled rows | Validation AUC | Public AUC | Outcome |
| --- | --- | --- | --- | --- | --- |
| B001 | [Initial baseline](baselines/001-catboost-gpu/README.md) | Kaggle T4 / 1,537,638 | 0.816026 | 0.80699 | Full-data baseline; 21 trees selected |
| T001 | [Timestamp split failure](trials/001-timestamp-split-failure/README.md) | Kaggle T4 / 1,537,638 | - | - | Failed before fitting; fixed in B001 |
| T002-A | [Summary-feature control](trials/002-embedding-expansion/README.md) | M1 CPU / 150,000 | 0.820875 | Not submitted | 679 features; 298 trees; 49.6 s fit |
| T002-B | [Embedding expansion](trials/002-embedding-expansion/README.md) | M1 CPU / 150,000 | 0.819478 | Not submitted | 1,127 features; 400 trees; 71.7 s fit |
| T003 | [Full GPU expanded vectors](trials/003-full-gpu-expanded-vectors/README.md) | Kaggle T4 / 1,537,638 | 0.816489 | Not submitted | +0.000463 vs B001; 22 trees; validation only |
| T004 | [Local TabM neural baseline](trials/004-local-tabm/README.md) | M1 Apple GPU / 150,000 | 0.809194 | Not submitted | Best epoch 1; overfit; 27.7 s fit |
| T005-A | [Custom regularized control](trials/005-custom-vector-network/README.md) | M1 Apple GPU / 150,000 | 0.813992 | Not submitted | Three branches; rare-ID pooling; 18.7 s fit |
| T005-B | [Custom vector attention](trials/005-custom-vector-network/README.md) | M1 Apple GPU / 150,000 | 0.814561 | Not submitted | +0.000569 vs control; 23.3 s fit |
| T006 | [Full GPU vector attention](trials/006-full-gpu-vector-attention/README.md) | Kaggle T4 / 1,537,638 | 0.827024 | 0.81905 (S002) | Epoch 1; +0.010535 vs T003; superseded by S003/S004 |
| T007-A | [Updated standalone TabM](trials/007-full-gpu-tabm-variants/README.md) | Kaggle T4 / 1,537,638 | 0.829229 | Not submitted | +0.002205 vs T006; 80.8 s fit |
| T007-B | [Attention encoder + TabM](trials/007-full-gpu-tabm-variants/README.md) | Kaggle T4 / 1,537,638 | 0.827934 | Not submitted | +0.000910 vs T006; epoch 1; 104.4 s fit |
| T008 | [Full TabM controls](trials/008-full-gpu-tabm-controls/README.md) | Kaggle T4 / 1,537,638 | **0.833419** | Not submitted | All 12 completed; removing high-cardinality fields wins; best at 7.74 epochs |
| T009 | [Full feature encodings](trials/009-full-gpu-feature-encodings/README.md) | Kaggle T4 / 1,537,638 | **0.835349** | Not submitted | All seven completed; exploratory vector crosses win |
| T010 | [Full tree/vector comparisons](trials/010-full-tree-vector-models/README.md) | Kaggle GPU/CPU / 1,537,638 | 0.825251 | Not submitted | All five completed; cross-feature CatBoost wins |
| T011 | [Full local LightGBM](trials/011-local-full-lightgbm/README.md) | Local CPU / 1,537,638 | 0.825584 | Not submitted | 10.6 min total; preliminary 75% T008 / 25% tree blend: **0.834336** |
| T012 | [LightGBM temporal backtest](trials/012-local-lightgbm-temporal-backtest/README.md) | Local CPU / 710,330 train + 299,969 valid | 0.818814 | Not submitted | Earlier fold, not directly comparable to primary AUC |
| T013 | [Existing-model ensembles](trials/013-existing-model-ensembles/README.md) | Local CPU / full primary holdout | **0.836899** | Not submitted | 21 unique models, 883 configurations; best equal mean 0.836669 |
| T014 | [Combinations and architectures](trials/014-full-gpu-combinations-architectures/README.md) | Kaggle T4 / 1,537,638 | **0.840454** | 0.83349 (S003) | All nineteen complete; PLR/cross/masked TabM k32 wins |
| T015 | [New cross-family ensembles](trials/015-tabm-member-ensembles/README.md) | Local CPU / full primary holdout | **0.840708** | Not submitted | 50% k32 + 25% k8 + 25% DCNv2-r32; four-family alternative 0.840670 |
| T016 | [Baseline CatBoost voting](trials/016-baseline-catboost-voting/README.md) | Local CPU / full primary holdout | **0.840806** | 0.83365 (S004) | 10% LightGBM + 90% T015; baseline CatBoost at 2.5% gives 0.840729; tiny gains uncertain |
| T017 | [Chronological OOF stacking](trials/017-full-temporal-stacking/README.md) | Kaggle GPU / local CPU / full temporal prefixes | **0.839489** | Not submitted | Beats matched fixed-budget neural (0.839237), below T016; all OOF rows, recovered without subsets |
| T018 | [Full seed/time confirmation](trials/018-full-seed-time-confirmation/README.md) | Kaggle GPU / both full temporal partitions | **0.840753** | Not submitted | All 18 fits complete; frozen seed-averaged 75/25 vote: primary 0.840753, earlier 0.837549; T016 remains best |
| T019 | [Relational history and GraphSAGE](trials/019-relational-history-graph/README.md) | Kaggle GPU / both full temporal partitions | 0.839237 control | Not submitted | All ten fits complete; history flat; graph recipe overfits (0.709957 primary); T016 remains best |
| T020 | [Local feature correlation audit](trials/020-local-feature-correlation/README.md) | M1 CPU / full 1,230,110-row training partition | Not a model | Not submitted | 1,104 numeric / 23 categorical features; 330 strongly correlated pairs; duplicate hour/category fields; target associations restricted before both outer holdouts |
| T021 | [Final full-data refit](trials/021-full-data-finalist-refit/README.md) | Kaggle GPU/CPU / all 1,537,638 labeled rows | No independent holdout | Not submitted | Complete and independently verified; both 1,037,090-row CSVs ready; frozen budgets and weights |
| T022 | [Saved-checkpoint sanity check](trials/022-checkpoint-former-holdout-sanity/README.md) | Inference only / former 307,528-row holdout | Not independent | Not submitted | Complete and verified; in-sample AUC: k32 0.846229, k8 0.847834, DCN 0.841304, LightGBM 0.851726, fixed ensemble 0.848533; no additional training |
| T023 | [Request/count/target-encoding screen](trials/023-request-count-target-screen/README.md) | Local LightGBM screen (scalar features) / original holdout | Holdout early stopping (optimistic) | Not submitted | Request context 0.828722 → 0.882612; counts 0.878542; target encoding no gain |
| T024 | [Request context in full models](trials/024-request-context-full-models/README.md) | T014/T011 protocols + 33 request features / original holdout | Holdout model selection (as T014) | S005/S006 | k32 0.889350, k8 0.888989, DCN 0.887607, LightGBM 0.880290; T016-weight ensemble 0.889809 (vs 0.840806) |
| T025 | [Request-context TabM k32 full refit](trials/025-request-tabm-k32-full-refit/) | Kaggle T4 / all 1,537,638 labeled rows | No independent holdout | 0.88467 (S007) | 26,629 fixed steps (17.73 epochs from T024); test rank correlation with S005 0.9938 |
| T026 | Wider request-context screen (`request_screen.py`) | Local LightGBM / original holdout | 0.883979 wide | Not submitted | rank/dev/gap for 51 gain-ranked fields beat control 0.882612; request mean/std no gain |
| T027 | [Wide request TabM k32](trials/027-wide-request-tabm-k32/) | Kaggle T4 / original holdout | **0.889850** | Not submitted | Best single model; +0.0005 vs T024 k32; best fractional epoch 8.985 |
| T028 | [Feed attention](trials/028-feed-attention/) | Kaggle T4 / original holdout | 0.889556 | Not submitted | Within-request attention; no gain over T027 (corr 0.995) |
| T029 | [Classic learners on wide features](trials/029-classic-wide-screen/) | Kaggle CPU / original holdout | 0.88728 XGBoost | Not submitted | 757 features; logreg OOM, CatBoost not run; 0.7 T027 + 0.3 XGB 0.89032 |
| T030 | [Date drift and item profiles](trials/030-date-drift-item-profiles/) | Kaggle CPU / original holdout | No gain | Not submitted | Dropping/normalising drifting fields and item profiles did not help |
| T031–T033 | CF, hurdle, residual/stack screens (`cf_screen.py`, `hurdle_screen.py`, `residual_screen.py`) | Local / original holdout | No gain | Not submitted | Residual on T027 0.88985 (no change); stacking raises within-request AUC but lowers global AUC |
| T034 | [XGBoost GPU tuning](trials/034-xgboost-gpu-tuning/) (`xgb_tune.py`) | Kaggle T4 / original holdout | 0.88849 | Not submitted | leaves511 eta .02 best; leaves63 0.88845, depth12 0.88842; native categorical 0.88532 |
| T035 | User × item cosine / kNN screen (`embed_screen.py`) | Local LightGBM / original holdout | 0.88488 | Not submitted | +0.0023 from 10 cosines with request rank/dev; embedding context and kNN video rates no extra gain |
| T036 | [XGBoost + cosines](trials/036-xgboost-cosines/) | Kaggle T4 / original holdout | **0.88967** depth12 | Not submitted | leaves63 0.88961, leaves511 0.88959; rank blend 0.45 T027 + .15/.15/.25 XGB **0.89081** |
| T037 | [Wide TabM k32 full refit](trials/037-wide-tabm-k32-full-refit/) | Kaggle T4 / all 1,537,638 labeled rows | No holdout | S008, S010 | Complete: 13,496 steps × seeds 42/43/44 (seed Spearman 0.997); seed-mean vs S007 Spearman 0.992 |
| T038 | [XGBoost + cosines full refit](trials/038-xgboost-cosines-full-refit/) | Kaggle T4 / all 1,537,638 labeled rows | No holdout | S008, S009 | Complete: 6 fits, rounds = best iteration × 1.25; candidates in `artifacts/candidates/` (best holdout-weighted 4-way blend 0.89081) |
| T039 | [XGBoost + cosines train-only test predictions](trials/039-xgboost-cosines-holdout-test/) | Kaggle T4 / original holdout fit rows | Same as T036 | Not submitted | Re-fits the 3 T036 configs (seed 42, best iteration) because T036 saved no models; reference only |
| T040 | [Wide architecture diversity](trials/040-wide-architecture-diversity/) | Kaggle T4 / original holdout | 0.88940 cross-masked-plr | Not submitted | fusion-cosine 0.88913, DCNv2-MoE 0.88883, FinalMLP gated 0.88854 / alternate 0.88845, DCNv2 0.88844; none enters the greedy blend |
| T041 | [CatBoost + pairwise XGBoost](trials/041-catboost-pairwise-xgboost/) | Kaggle T4 / original holdout | 0.88829 CatBoost+IDs | Not submitted | CatBoost 0.88821, pairwise XGBoost 0.88586; greedy blend with CatBoost+IDs 0.89109 (+0.00008) |
| T042 | DeepFM with ID embeddings (`deepfm_screen.py`) | Local MPS / original holdout | 0.88426 | Not submitted | FM over 21 categorical/ID fields + 16 numeric fields, 512-256-128 MLP |
| T043 | ModernNCA-style retrieval (`nca_screen.py`) | Local MPS / original holdout | 0.87923 | Not submitted | Softmax(-distance) neighbour-label model over learned embeddings |
| T044 | Exposure-graph SVD (`svd_screen.py`) | Local LightGBM / original holdout | Running | Not submitted | Train+test user×item and user×creator degree-normalised SVD (k=32): factors, affinity, request rank/dev |
| T045 | [CatBoost + IDs full refit](trials/045-catboost-ids-full-refit/) | Kaggle T4 / all 1,537,638 labeled rows | No holdout | S011, S012 | 2925 iterations × seeds 42/43/44; Complete (seed Spearman 0.988); completes the 5-model blend (holdout 0.89102) |
| T046 | [GBDT 5-fold](trials/046-gbdt-5fold-full/) | Kaggle T4 / GroupKFold(5) by request, all labeled rows | OOF xgbd12 0.88902, xgb511 0.88889, catboost_ids 0.88742 | Candidates | Complete; fold-mean vs full-refit Spearman 0.996-0.999 |
| T047 | [Wide TabM k32 5-fold](trials/047-wide-tabm-k32-5fold/) | Kaggle T4 / same 5 folds | OOF 0.88893 (folds 0.8870-0.8916) | Candidates | Complete, 10,800 steps per fold; OOF 4-family blend 0.89007 equal / 0.89019 TabM×2 |
| T048 | [TabM listwise + capacity](trials/048-tabm-listwise-capacity/) | Kaggle T4 / holdout | Best w256-b3 0.88978 | Rejected | Blend ≤ 0.89113 vs base 0.89109 |
| T049 | [XGBoost entity features](trials/049-xgboost-entity-features/) | Kaggle T4 / holdout | — | Rejected | No blend gain |
| T050 | [Multi-tower](trials/050-multitower/) | Kaggle T4×2 / holdout | content-only 0.88479 | Rejected | In-sample GBDT leaves overfit; blend 0.89098 < base |
| — | TabICL screen (private kernel) | T4, 50k context, 30k holdout rows | 0.88052 | Rejected | Blend P(gain>0)=0.055; 80k context OOM |
| — | [Horizon drift screen](horizon_screen.py) | LightGBM fit first 32h, scored 0–40h ahead | 0.870→0.873 by horizon | Rejected | No horizon decay; drop-3 gain (+0.005) was native-categorical overfit, +0.0004 with numeric codes |
| — | Duration proxy | feature54 = duration seconds (45–300) | watch rate 7.5%→0.07% | Rejected | Already used raw + request rank/dev/gap |
| S001 | [Submitted B001](submitted/001-catboost-gpu/README.md) | Full-data refit | 0.816026 | 0.80699 | Submission 58271142; Kaggle notebook version 2 |
| S002 | [Submitted T006 checkpoint](submitted/002-vector-attention/README.md) | Unchanged epoch-1 checkpoint | 0.827024 | **0.81905** | Submission 58275874; no refit; +0.01206 vs S001 |
| S003 | [Submitted best TabM k32](submitted/003-tabm-k32/README.md) | Unchanged T014 seed-42 checkpoint | **0.840454** | **0.83349** | Submission 58291755; no refit; +0.01444 vs S002 |
| S004 | [Submitted best frozen ensemble](submitted/004-best-ensemble/README.md) | Unchanged T014/T011 checkpoints | **0.840806** | **0.83365** | Submission 58291773; no refit; +0.00016 vs S003 |
| S005 | [Submitted request-context TabM k32](submitted/005-request-tabm-k32/README.md) | Unchanged T024 k32 checkpoint | **0.889350** | **0.88394** | Submission 58300770; no refit; +0.05045 vs S003 |
| S006 | [Submitted request-context ensemble](submitted/006-request-ensemble/README.md) | Unchanged T024 checkpoints, T016 weights | **0.889818** | **0.88438** | Submission 58300782; no refit; +0.05073 vs S004 |
| S007 | [Submitted request-context TabM k32 full refit](submitted/007-request-tabm-k32-full-refit/README.md) | T025 full-data refit (1,537,638 rows) | No holdout | **0.88467** | Submission 58304230; +0.00073 vs S005 |
| S008 | [Submitted full-refit 4-way blend](submitted/008-full-refit-4way-blend/README.md) | T037 seed mean + T038 XGBoost, holdout rank weights .45/.15/.15/.25 | **0.89081** | **0.88641** | Submission 58332687; +0.00174 vs S007; best public score |
| S009 | [Submitted full-refit XGBoost blend](submitted/009-full-refit-xgboost-blend/README.md) | T038 XGBoost, rank weights .3/.3/.4 | 0.89001 | 0.88579 | Submission 58332848; diagnostic |
| S010 | [Submitted full-refit wide TabM seed mean](submitted/010-full-refit-wide-tabm-seed-mean/README.md) | T037 3-seed mean | 0.88985 | 0.88533 | Submission 58332854; diagnostic |
| S011 | [Submitted full-refit 5-way blend + CatBoost-IDs](submitted/011-full-refit-5way-cbids/README.md) | Equal rank: T037 + T025 + T038 xgb511/xgbd12 + T045 | 0.89102 | 0.88646 | Submission 58354017; best |
| S012 | [Submitted full-refit CatBoost-IDs](submitted/012-full-refit-catboost-ids/README.md) | T045 3-seed mean | 0.88829 | 0.88388 | Submission 58354023; signal run |
| S013 | [5-fold+full 5-way equal](submitted/013-5way-kfold-equal/README.md) | S011 full refits + T046/T047 5-fold means, equal rank | OOF n/a | **0.88647** | Submission 58451223; best (+0.00001) |
| S014 | [5-fold+full 5-way TabM×2](submitted/014-5way-kfold-tabm2/README.md) | As S013, TabM ×2 | OOF 0.89019 (4-family) | 0.88647 | Submission 58451231 |
| S015 | [XGBoost-heavy 5-way](submitted/015-5way-xgb60/README.md) | XGB 60% | — | 0.88637 | Submission 58451238 |
| S016 | [5-way TabM×2 + CatBoost-IDs](submitted/016-5way-tabm2-cbids/README.md) | S011 components, TabM ×2 | — | 0.88645 | Submission 58451245; blend saturated |

T002-A and T002-B share the same sampled rows and holdout and are directly
comparable. Do not infer that either beats B001: B001 used a different, full-data
holdout and GPU training. Public scores and local validation scores are different
measurements.

The local subset misranked CatBoost summaries and custom attention: T002-A beat
T005-B locally, but T006 beat B001 on the full holdout. For learning curves, fix
the full chronological validation cohort and sample only training rows; do not
treat a subset ranking as a strict gate for rejecting a model family.

## Layout

- `baselines/`: initial reference notebook, configuration, metrics, and README.
- `trials/`: numbered experiments, each with a notebook, README, and recorded results.
- `submitted/`: immutable source/notebook snapshots and scoreboard records for actual submissions.
- `train.py`: shared training and prediction pipeline; defaults to summary features.
- `experiments.py`: paired local summary-versus-expanded-embedding experiment.
- `tabm_experiment.py`: optional PyTorch/TabM neural subset experiment.
- `custom_network.py`: paired regularized three-branch / vector-attention networks.
- `tabm_variants.py`: updated standalone / attention-hybrid TabM on a fixed holdout.
- `full_experiments.py`: full-partition ablation batches with step-level checkpoints.
- `predict_custom.py`: checkpoint-only inference with environment, replay, and ID checks.
- `download.py`: authenticated, resumable competition download.
- `data/`: local competition files; ignored by Git.
- `artifacts/`: local models, cached samples, and predictions; ignored by Git.

## Setup and local experiments

Requires Python 3.12 or 3.13 and `uv`.

```bash
uv sync --locked
uv run python download.py
uv run python experiments.py --rows 150000 --iterations 400 --threads 4 \
  --output-dir artifacts/local-v1
```

The downloader reads `~/.kaggle/access_token`; credentials never belong in this
repository. Competition access and acceptance of its rules are required.

To open the tracked notebooks:

```bash
uv sync --extra notebooks
uv run jupyter lab
```

For a standalone local baseline with prediction generation:

```bash
uv run python train.py --rows 150000 --task-type CPU \
  --output-dir artifacts/baseline-local
```

Local defaults use a seed-42 uniform 150,000-row sample. Training and validation
are split chronologically 80/20 by `feature1`, keeping timestamp ties together.
Kaggle defaults select all 1,537,638 labeled rows and require a CUDA GPU.

Optional local deep-learning trial and its focused checks:

```bash
uv sync --locked --extra deep-learning
uv run --extra deep-learning python -m unittest discover -s tests -q
uv run --extra deep-learning python tabm_experiment.py --rows 150000
```

TabM selects CUDA, Apple MPS, or CPU in that order and reports its device.
Use `--device cpu`, `--device mps`, or `--device cuda` to require a specific backend.
It only validates and saves the best checkpoint; it never submits predictions.

| Dataset | Rows | Bytes |
| --- | ---: | ---: |
| `train.parquet` | 1,537,638 | 3,458,726,237 |
| `test.parquet` | 1,037,090 | 2,470,609,389 |
| `sample_submission.csv` | 1,037,090 | 11,333,981 |

## Recording future trials

Create `trials/NNN-short-description/` before a new experiment. Keep a notebook
and one README alongside it, describing the hypothesis, exact configuration,
data selection, seed, split, source revision, measured results, runtime, and
decision. Save small machine-readable metrics and relevant logs in that folder;
keep large outputs in ignored `artifacts/`. Add a row to the ledger above.

Reuse the versioned Kaggle validation notebook
`dheerajgattupalli/sharechat-model-experiments-gpu` (display title: ShareChat Model
Experiments GPU) for future training trials instead of creating one Kaggle
notebook per trial. Preserve each executed version in that trial's provenance.
Keep historical Git snapshots and submitted-model dependency notebooks unchanged.
This is the renamed original expanded-vectors notebook: version 1 is T003 and
version 2 is T007. Renaming did not create an additional Kaggle notebook.

Preserve old results instead of overwriting them. Reproduce historical trials
from their recorded source revision or frozen notebook, not an arbitrarily newer
shared pipeline. Environment locks are included for local reproducibility;
the original Kaggle image digest is recorded in the archived kernel metadata,
but its exact package versions were not enumerated. GPU fits need not be
bit-for-bit reproducible.

## Kaggle training and submission policy

Verify code correctness locally before launching full-data GPU runs. Compare
models on the same full chronological holdout; sample training only for smaller
learning-curve experiments.
Use held-out results to select model capability, feature recipes and training
budgets. Only after selecting the final approach, refit on all labeled rows to
generate submission predictions; that refit has no independent holdout score.
The root `kernel-metadata.json` describes the private Kaggle GPU script and
attaches the `sharechat` competition data with internet disabled.

**Never make another competition submission without explicit user approval for
that submission.** Training or generating predictions is not submission approval.
No training script or notebook submits predictions automatically.

For an approved submission, first archive the exact executed notebook/source,
configuration, validation metrics, prediction-file SHA-256, Kaggle version, and
submission ID in a new `submitted/NNN-description/` folder. Add the public score
when grading completes. Submission archives must not be edited into newer models.
