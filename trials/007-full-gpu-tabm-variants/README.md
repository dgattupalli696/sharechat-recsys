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

| Model | Validation ROC AUC | Delta vs T006 | Best epoch | Training wall time |
| --- | ---: | ---: | ---: | ---: |
| T006: custom vector attention | 0.8270238904 | - | 1 | 98.91 s |
| T007-A: updated standalone TabM | **0.8292287153** | **+0.0022048249** | 1 | 80.83 s |
| T007-B: attention encoder + TabM | 0.8279340616 | +0.0009101712 | 1 | 104.37 s |

Standalone TabM is the best full-data validation result so far. It also exceeds
the hybrid by **0.0012946537 AUC**. Both models completed six epochs, restored
their first-epoch checkpoints, and still showed overfitting: epoch-6 AUC fell
to 0.8108963085 for standalone and 0.8080107397 for hybrid.

Shared preparation took **281.82 seconds**. Standalone has 770,216 parameters;
the hybrid has 773,392. The 971 dense preprocessed channels plus 476 vector
channels form 1,447 numeric inputs. Standalone adds 184 category-embedding
channels for 1,631 TabM inputs; the hybrid feeds its 256 fused branch outputs
into its TabM head.

The fitted preprocessor is byte-identical to T006's saved artifact. Saved
validation labels/order exactly match T006, and independent AUC recomputation
from both prediction files reproduces the reported scores. Checkpoint,
preprocessor, prediction and report hashes are recorded in `provenance.json`.

This is a single-seed exploratory comparison, not a significance claim. The
old T004 subset score of 0.8091940040 used a smaller training set and different
holdout/features/regularization, so its score difference cannot be attributed
solely to these preprocessing changes. No public leaderboard score is generated.

## Versioning and reproduction

Reused the existing expanded-vectors notebook and renamed its display title to
**ShareChat Model Experiments GPU**. Kaggle consequently changed its slug to
`dheerajgattupalli/sharechat-model-experiments-gpu`; the notebook identity remains
144588605. No additional Kaggle notebook was created.

Version 1 still exactly matches the frozen T003 source; version 2 is this T007
pair. `kernel-metadata.json` preserves the original launch request with the old
slug. The exported executed metadata records the canonical new slug. Do not
modify the separate T006 notebook on which submitted S002 depends.

Source revision: `a48b4f06715dd02cd9667be4b618e2436366134e`.

The notebook freezes all four source files and offline wheels. Version 2
completed successfully; exported source cells exactly match the frozen notebook.
Original `results.json`, `run.log` and `executed-kernel-metadata.json` are archived
here; large artifacts stay in ignored `artifacts/kaggle-t007/`.

Kaggle versions match T006 exactly: PyTorch 2.11.0+cu130, NumPy 2.1.3,
Polars 1.35.2, scikit-learn 1.6.1; additional packages are TabM 0.0.3 and
rtdl-num-embeddings 0.0.12. The pinned image digest is recorded in executed
metadata and provenance.

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
