# ShareChat RecSys Challenge

Predict the probability that a user watches a video. Competition metric: ROC AUC.

## Run ledger

All runs below were performed on November 12, 2026. A trial is not a leaderboard
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
| T006 | [Full GPU vector attention](trials/006-full-gpu-vector-attention/README.md) | Kaggle T4 / 1,537,638 | 0.827024 | 0.81905 (S002) | Best submitted model; epoch 1; +0.010535 vs T003 |
| T007-A | [Updated standalone TabM](trials/007-full-gpu-tabm-variants/README.md) | Kaggle T4 / 1,537,638 | **0.829229** | Not submitted | Best full-data validation; +0.002205 vs T006; 80.8 s fit |
| T007-B | [Attention encoder + TabM](trials/007-full-gpu-tabm-variants/README.md) | Kaggle T4 / 1,537,638 | 0.827934 | Not submitted | +0.000910 vs T006; epoch 1; 104.4 s fit |
| T008 | [Full TabM controls](trials/008-full-gpu-tabm-controls/README.md) | Kaggle T4 / all rows | Pending | Not submitted | 12 full-data checkpoint/optimization/category runs; no subset elimination |
| T009 | [Full feature encodings](trials/009-full-gpu-feature-encodings/README.md) | Kaggle T4 / all rows | Pending | Not submitted | PLE, PLR, quantiles, vector projection/dropout/crosses; all full runs |
| S001 | [Submitted B001](submitted/001-catboost-gpu/README.md) | Full-data refit | 0.816026 | 0.80699 | Submission 58271142; Kaggle notebook version 2 |
| S002 | [Submitted T006 checkpoint](submitted/002-vector-attention/README.md) | Unchanged epoch-1 checkpoint | 0.827024 | **0.81905** | Submission 58275874; no refit; +0.01206 vs S001 |

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
The root `kernel-metadata.json` describes the private Kaggle GPU script and
attaches the `sharechat` competition data with internet disabled.

**Never make another competition submission without explicit user approval for
that submission.** Training or generating predictions is not submission approval.
No training script or notebook submits predictions automatically.

For an approved submission, first archive the exact executed notebook/source,
configuration, validation metrics, prediction-file SHA-256, Kaggle version, and
submission ID in a new `submitted/NNN-description/` folder. Add the public score
when grading completes. Submission archives must not be edited into newer models.
