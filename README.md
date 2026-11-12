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
| T004 | [Local TabM neural baseline](trials/004-local-tabm/README.md) | M1 / 150,000 | Pending | Not submitted | Summary features; neural ensemble; validation only |
| S001 | [Submitted B001](submitted/001-catboost-gpu/README.md) | Full-data refit | 0.816026 | 0.80699 | Submission 58271142; Kaggle notebook version 2 |

T002-A and T002-B share the same sampled rows and holdout and are directly
comparable. Do not infer that either beats B001: B001 used a different, full-data
holdout and GPU training. Public scores and local validation scores are different
measurements.

## Layout

- `baselines/`: initial reference notebook, configuration, metrics, and README.
- `trials/`: numbered experiments, each with a notebook, README, and recorded results.
- `submitted/`: immutable source/notebook snapshots and scoreboard records for actual submissions.
- `train.py`: shared training and prediction pipeline; defaults to summary features.
- `experiments.py`: paired local summary-versus-expanded-embedding experiment.
- `tabm_experiment.py`: optional PyTorch/TabM neural subset experiment.
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

Preserve old results instead of overwriting them. Reproduce historical trials
from their recorded source revision or frozen notebook, not an arbitrarily newer
shared pipeline. Environment locks are included for local reproducibility;
the original Kaggle image digest is recorded in the archived kernel metadata,
but its exact package versions were not enumerated. GPU fits need not be
bit-for-bit reproducible.

## Kaggle training and submission policy

Verify changes on a local subset before launching promising full-data GPU runs.
The root `kernel-metadata.json` describes the private Kaggle GPU script and
attaches the `sharechat` competition data with internet disabled.

**Never make another competition submission without explicit user approval for
that submission.** Training or generating predictions is not submission approval.
No training script or notebook submits predictions automatically.

For an approved submission, first archive the exact executed notebook/source,
configuration, validation metrics, prediction-file SHA-256, Kaggle version, and
submission ID in a new `submitted/NNN-description/` folder. Add the public score
when grading completes. Submission archives must not be edited into newer models.
