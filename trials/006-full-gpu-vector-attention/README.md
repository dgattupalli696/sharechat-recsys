# T006 - Full-data GPU custom vector attention

Run date: November 12, 2026. **One attention-model run, validation only; no submission.**

## Hypothesis and configuration

Scale the T005-B custom vector-attention network from the 150,000-row subset to
all 1,537,638 labeled rows, without changing its architecture or learning settings.

- Exact full-data chronological holdout used by B001/T003: 1,230,110 training rows,
  307,528 validation rows; cutoff October 28, 2026, 08:06:52.397 in dataset time.
- Input information: all 1,127 summary-plus-expanded engineered features.
- Same three branches, per-field vector projections, learned missing tokens,
  and single four-head 32-channel vector-attention block as T005-B.
- Same training-only preprocessing: medians, missing indicators, standardization
  and clipping; category minimum count 10, eight-dimensional embeddings,
  category dropout 0.3; vector direction, log magnitude, and presence indicators.
- Same 128/64/64 branch outputs and 64-unit fusion head. Training-derived category
  vocabulary sizes and missingness indicators may change with the larger dataset.
- AdamW learning rate 0.0005, weight decay 0.01; batch size 1,024; maximum 30
  epochs, patience 5; seed 42, four CPU threads.
- Compute: private Kaggle T4 notebook; explicit CUDA requirement; internet disabled.
- Output: fitted preprocessor, best validation checkpoint, validation probabilities,
  training history, and metrics only. No full-data refit or test prediction.

## Full-data execution details

Numerical preprocessing now works sequentially on training then validation arrays
and reuses buffers. Vector channels are appended one field at a time into a
preallocated tensor. Chronological partitions are zero-copy slices of sorted rows,
with the split position found at the first timestamp equal to the cutoff. These
are memory optimizations; transformation semantics and timestamp-tie handling are
unchanged.

The notebook freezes all definitions and attaches the competition files directly.
It uses no pretrained model weights and does not require the TabM package:
the reused training helpers are independent of TabM construction, which is now
imported lazily only when that architecture is instantiated.

## Results

Pending Kaggle execution. Comparable full-data reference AUCs:

| Model | Validation ROC AUC |
| --- | ---: |
| B001: CatBoost summaries | 0.8160260803 |
| T003: CatBoost expanded vectors | 0.8164893363 |

T005-B's 0.8145605736 used a different, smaller holdout and is not directly
comparable. One GPU run cannot establish statistical significance.

## Reproduction and tracking

Kaggle kernel: `dheerajgattupalli/sharechat-custom-vector-attention-gpu`.
Source and configuration hashes are recorded in `provenance.json`. Results and
the original execution log will be archived beside this README when the run
finishes; large artifacts stay in ignored `artifacts/kaggle-t006/`.

The equivalent CLI, on a machine with CUDA and the competition data, is:

```bash
uv run --extra deep-learning python custom_network.py \
  --data-dir /path/to/competition/data --rows 1600000 \
  --variant attention --device cuda --output-dir artifacts/kaggle-t006
```

The cap selects all available labeled rows. This performs validation only and
never submits predictions. The frozen notebook passes the same options using
Kaggle's attached input and `/kaggle/working/` output directories.
