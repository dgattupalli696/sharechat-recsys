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

Pending Kaggle execution. The reference B001 validation ROC AUC is 0.8160260803.
Compare against that value, not against the differently sized local T002 holdout.

## Tracking

Kaggle kernel: `dheerajgattupalli/sharechat-expanded-vectors-gpu`.
Source and configuration hashes are recorded in `provenance.json`. Logs and
metrics will be archived beside this README after execution. Large artifacts
remain in ignored `artifacts/kaggle-t003/`.

Reproduction through the shared CLI:

```bash
uv run python train.py --rows 1600000 --task-type GPU \
  --expand-embeddings --validation-only --output-dir artifacts/kaggle-t003
```

Running this command on a Mac without CUDA must use `--task-type CPU` instead.
Neither command submits predictions to Kaggle.
