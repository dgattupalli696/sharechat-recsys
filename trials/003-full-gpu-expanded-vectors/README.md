# T003 - Full-data GPU training with expanded vectors

Run date: November 12, 2026. **Validation only; no competition submission.**

## Hypothesis

T002's 150,000-row CPU experiment did not benefit from expanded vectors. Test
whether retaining individual vector values helps when training on the full
dataset using the same GPU configuration and chronological holdout as B001.

## Configuration

- Dataset: all 1,537,638 labeled rows; seed 42.
- Split: 1,230,110 training rows and 307,528 validation rows.
- Cutoff: October 28, 2026, 08:06:52.397 in dataset timestamp representation.
- Features: B001's 679 features plus 448 vector dimensions, for 1,127 total.
  Keep the original summary features; reject non-null vectors not of length 32.
- Model: CatBoost GPU, depth 6, learning rate 0.06, Logloss objective, AUC
  evaluation, maximum 400 trees, early stopping patience 60,
  `max_ctr_complexity=1`, four CPU threads.
- Compute: private Kaggle T4 notebook, competition data attached, internet disabled.
- Output: validation model, metrics, and logs only. No full-data refit, test
  predictions, or submission is needed to answer this experiment.

The notebook freezes the training implementation and explicitly passes
`--expand-embeddings --validation-only`. Locally it uses the repository's `data/`
directory and CPU; on Kaggle it requires a CUDA GPU. Its row cap is 1,600,000,
so both environments select all available labeled rows.

## Results

| Run | Features | Selected trees | Validation ROC AUC |
| --- | ---: | ---: | ---: |
| B001: summary baseline | 679 | 21 | 0.8160260803 |
| T003: summaries + expanded vectors | 1,127 | 22 | 0.8164893363 |

Change: **+0.0004632559 AUC** on the exact same 307,528-row chronological holdout.
Expanded vectors helped slightly in this full-data GPU run, unlike the smaller
CPU experiment. The gain is small and comes from one GPU run per configuration;
there are no repeated seeds or confidence intervals, and GPU training is
nondeterministic. Do not treat this as an established leaderboard improvement.

The fit call took 54.08 seconds, including model setup. The training loop itself
was approximately eight seconds before early stopping. The log reaches completed
validation and the validation-only exit at approximately 128 seconds from kernel
start. No refit or test prediction was performed.

Both runs used the same Kaggle image digest, GPU model configuration, labeled
rows, and timestamp cutoff. Compare against B001, not the differently sized
local T002 holdout. **No competition submission was made.**

## Tracking

Kaggle kernel: `dheerajgattupalli/sharechat-expanded-vectors-gpu`, version 1.
Source revision: `f765854c8e418a70c78061965fafa30fc600a79c`.

Source and configuration hashes are recorded in `provenance.json`. Original
validation metrics, the execution log, and exported Kaggle environment metadata
are archived here as `metrics.json`, `run.log`, and
`executed-kernel-metadata.json`. The validation model remains in ignored
`artifacts/kaggle-t003/`; its checksum is recorded in provenance.

Reproduction through the shared CLI:

```bash
uv run python train.py --rows 1600000 --task-type GPU \
  --expand-embeddings --validation-only --output-dir artifacts/kaggle-t003
```

Running this command on a Mac without CUDA must use `--task-type CPU` instead.
Neither command submits predictions to Kaggle.
