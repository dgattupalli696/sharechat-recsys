# T007 - Full-data updated TabM and attention-TabM

Run date: November 12, 2026. **Two validation-only models; no submission.**

## Hypothesis

Revisit TabM using the newer T006 preprocessing and regularization, rather than
rejecting it based on the old 150,000-row T004 run. Compare two ways of using
the same input information on exactly the full B001/T003/T006 chronological holdout.

- T007-A, standalone: normalized numeric features, vector summaries, flattened
  normalized raw vectors and learned categorical embeddings feed TabM directly.
- T007-B, hybrid: retain T006's scalar, categorical and vector-attention branches,
  but replace its final MLP head with an eight-member TabM ensemble receiving
  the 256 fused branch channels. Train the entire network jointly from scratch;
  do not freeze or reuse the submitted model's weights.

These are exploratory configurations, not the paper's fully tuned models.
Changes relative to T004 include training size, vector inputs and regularization;
this pair cannot isolate each change's individual effect.

## Shared configuration

- All 1,537,638 labeled rows; 1,230,110 training and 307,528 validation.
- Chronological cutoff: October 28, 2026, 08:06:52.397 in dataset time.
  Keep all timestamp ties together. Never use test data.
- Exactly T006's train-only imputation, missingness indicators, standardization
  and clipping; category minimum count 10; eight-dimensional category embeddings;
  training-only category dropout 0.3.
- Each vector has 32 direction coordinates, clipped log magnitude and a presence
  indicator. Keep summaries too. Nonfinite vectors are missing; zero vectors
  remain present. The hybrid additionally uses the original learned missing tokens,
  per-field projections and four-head attention.
- TabM 0.0.3: eight members, two 128-unit blocks, dropout 0.2.
  Initialize each member's output bias from the training positive rate.
- AdamW learning rate 0.0005, weight decay 0.01; batch size 1,024; maximum
  30 epochs, patience five; seed 42, four CPU threads.
- Train each member with binary cross-entropy; average probabilities at evaluation.
  Reload the best checkpoint and verify its selected AUC.
- One shared fitted preprocessor and exactly identical validation rows/order for
  both models. Save labels, probabilities, histories and separate checkpoints.
- Private Kaggle T4, same pinned image as T006, internet disabled.
  Two small unmodified, licensed pure-Python wheels are embedded in the notebook,
  with SHA-256 verified against `uv.lock`: TabM 0.0.3 and
  rtdl-num-embeddings 0.0.12. Install without dependencies or internet, preserving
  the image's PyTorch and scientific stack and the packages' original licenses.

## Results

Pending full-data execution. No public leaderboard score is requested or generated.

## Versioning and reproduction

Reuse `dheerajgattupalli/sharechat-expanded-vectors-gpu`, with the general display
title **ShareChat Model Experiments GPU**; do not create another Kaggle notebook.
Version 1 remains the original T003 experiment. Record the newly executed
version and source commit in `provenance.json` after launch. Do not modify the
separate T006 notebook on which submitted S002 depends.

The notebook freezes all four source files and offline wheels. Original
results/logs and executed metadata will be archived here; large artifacts stay
in ignored `artifacts/kaggle-t007/`.

```bash
uv sync --locked --extra deep-learning
uv run --extra deep-learning python tabm_variants.py \
  --data-dir /path/to/competition/data --device cuda \
  --output-dir artifacts/kaggle-t007
```

The runner reads the complete dataset before determining its chronological
holdout. For subsequent local learning curves, add `--training-rows 150000`:
only the earlier training partition is sampled; the validation rows and cutoff
are unchanged. `--sample-path` instead replays an already engineered dataset
and defines the holdout from that file; a subset file is not the full holdout.

Training/inference artifacts are not competition submissions. Both runs are
validation only: no refit, test predictions, or submission API calls.
