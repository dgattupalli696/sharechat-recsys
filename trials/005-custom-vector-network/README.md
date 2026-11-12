# T005 - Custom regularized recommendation networks

Run date: November 12, 2026. **Paired local validation; no competition submission.**

## Motivation

T004's first neural baseline overfit sharply after epoch 1. In its chronological
holdout, 98.1% of `feature5` values were absent from the training vocabulary.
Memorizing high-cardinality IDs is unlikely to generalize to that cold-start
setting, although T004 did not establish the cause of overfitting.

Build two small, matched networks designed around the actual input types rather
than treating every value as an interchangeable scalar. This is a custom
combination of established ideas, not a claim of a new research algorithm.

## Shared design

```text
scalar features + vector summaries -> impute/scale/missing flags -> 128 channels
categorical fields -> rare-ID pooling + embedding dropout       ->  64 channels
14 raw 32-value vectors -> separate field projections           ->  64 channels
                         concatenate -> small prediction head -> watch probability
```

- Exact cached T002/T004 sample: 150,000 labeled rows, seed 42.
- Chronological holdout: 119,999 training and 30,001 validation rows, cutoff
  October 28, 2026, 08:06:29.680 in dataset timestamp representation.
- Input information: all 1,127 summary-plus-expanded engineered features.
  The same numeric preprocessing as T004 is fit only on training rows.
- Categories: keep levels occurring at least 10 times in training. All rarer or
  unseen levels use a reserved zero-vector entry. Each retained category has an
  eight-dimensional learned embedding; drop entire categorical entries with
  probability 0.3 during training.
- Vectors: normalize direction per row, retaining clipped log magnitude and a
  presence flag. Zero vectors remain present. Vectors with any nonfinite
  coordinate become missing fields, with counts recorded. Separate 33-to-32
  projections avoid assuming that different vector fields share coordinates or
  meaning. Learn field identifiers and missing-vector tokens.
- Scalar and categorical branches are small normalized MLPs. Vector tokens share
  a residual per-token MLP and a small flattened fusion layer.
- Fuse the three branches through a 64-unit head. Initialize only the output bias
  from the training positive rate. No reweighting or resampling.
- AdamW: learning rate 0.0005, weight decay 0.01; batch size 1,024; maximum
  30 epochs with patience 5; seed 42, four CPU threads.
- Train on Apple MPS when available. Both variants use the same preprocessing,
  parameter initialization for shared modules, and seed-42 minibatch ordering.

## Controlled variants

T005-A, `control`: the three-branch model without a dedicated attention block.
Its fusion MLP can still learn interactions.

T005-B, `attention`: add one 32-channel, four-head attention block with
pre-normalization and residual dropout between the 14 vector tokens. All other
architecture and training settings stay the same.

We deliberately do not call the vector groups "users" or "videos": those
semantics are not established by the anonymized column names. Missing fields are
learned tokens, not fully masked attention rows, so an all-missing input remains
numerically defined.

## Results

| Trial | Validation ROC AUC | Selected training point | Fit wall time |
| --- | ---: | --- | ---: |
| T002-A: CatBoost summaries, CPU | 0.8208754309 | 298 trees | 49.61 s |
| T002-B: CatBoost expanded vectors, CPU | 0.8194781205 | 400 trees | 71.72 s |
| T004: TabM summaries, Apple GPU | 0.8091940040 | Epoch 1 | 27.66 s |
| T005-A: custom control, Apple GPU | 0.8139920305 | Epoch 2 | 18.75 s |
| T005-B: custom attention, Apple GPU | 0.8145605736 | Epoch 2 | 23.30 s |

Both custom models stopped after seven epochs and restored epoch 2. Shared
preparation took 14.96 seconds. The control has 276,665 parameters; attention adds
4,288, for 280,953 total.

Attention changed AUC by **+0.0005685431** versus the matched custom control,
and by **+0.0053665695** versus T004. It remains **0.0063148573 below T002-A**.
These are single exploratory runs, not statistically established improvements.
There is no full-data GPU result or public leaderboard score for either model.

The numerical branch received 971 channels after train-only missingness
indicators. Rare pooling removed all learnable `feature5` category IDs in this
subset: its 95,210 training levels became the reserved unknown entry. Other
categorical vocabulary sizes are recorded in the results. This is a deliberate
cold-start defense, not a claim that the field is globally useless.

Overfitting was reduced relative to T004's training-loss collapse but not
eliminated. Both models' holdout AUC declined after epoch 2. Raw vector
interactions and stronger regularization alone did not beat the tree baseline.

T005-A versus T005-B isolates the added attention block. Comparisons against
earlier trials also change preprocessing, regularization, optimizer, and/or
vector representation; they do not isolate architecture alone. Runtime
comparisons against T002 also use different compute backends.

## Reproduction and tracking

```bash
uv sync --locked --extra deep-learning --extra notebooks
uv run --extra deep-learning python custom_network.py \
  --sample-path artifacts/local-v1/sample.parquet --rows 150000 \
  --device mps --output-dir artifacts/local-custom-v1
```

Omit `--sample-path` to regenerate the same seed-42 rows with expanded vector
features from `data/train.parquet`. Use `--device cpu` when Apple MPS is absent.
`--variant control` or `--variant attention` runs a single variant; default is both.

The notebook freezes feature preparation, the reused neural training/checkpoint
helpers, and the custom model/runner. Original model, preprocessor and validation
prediction files remain under ignored `artifacts/local-custom-v1/`; small results,
logs, environment and source/data/artifact checksums are archived here.

Source revision: `08e1b805f09aa5c9e614a5d5b6c331ac117d4dfa`.
The original cached sample SHA-256 matches T004. Both checkpoints were reloaded;
their saved validation probabilities independently reproduce the recorded AUCs.
Environment: Python 3.12, PyTorch 2.15.0 on Apple MPS, NumPy 2.5.4, Polars 1.44.2,
scikit-learn 1.9.1. GPU operations can be nondeterministic.

Original T004 notebooks remain frozen; extracting shared training code does not
alter the historical trial. Neither this trial nor its notebooks run Kaggle
compute, generate test predictions, or submit to the competition.
