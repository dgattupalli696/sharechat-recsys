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

Pending local execution. References on this subset:

| Prior trial | Validation ROC AUC |
| --- | ---: |
| T002-A: CatBoost summaries | 0.8208754309 |
| T002-B: CatBoost expanded vectors | 0.8194781205 |
| T004: TabM summaries | 0.8091940040 |

T005-A versus T005-B isolates the added attention block. Comparisons against
earlier trials also change preprocessing, regularization, optimizer, and/or
vector representation; they do not isolate architecture alone.

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
logs, environment and source/data/artifact checksums will be archived here.

Original T004 notebooks remain frozen; extracting shared training code does not
alter the historical trial. Neither this trial nor its notebooks run Kaggle
compute, generate test predictions, or submit to the competition.
