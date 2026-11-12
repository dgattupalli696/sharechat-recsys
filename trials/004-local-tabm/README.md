# T004 - Local TabM neural baseline

Run date: November 12, 2026. **Local subset validation only; no competition submission.**

## Hypothesis

Try a neural tabular learner rather than another tree model. TabM uses
parameter-efficient ensembles of neural networks. The first experiment uses the
same summary-based input features as T002-A so the vector representation does
not change at the same time as the algorithm.

## Configuration

- Data: the exact cached seed-42 150,000-row sample from T002.
- Holdout: 119,999 training rows and 30,001 validation rows, split chronologically
  at October 28, 2026, 08:06:29.680 in dataset timestamp representation.
- Inputs: 679 summary-based features, including 23 categorical columns.
- Numeric preprocessing: training-only median imputation, zero for all-missing
  columns, missingness indicators, standardization, clipping to [-10, 10].
  Infinite values, if encountered, are explicitly reported and treated as missing.
- Categories: training-only vocabulary; eight-dimensional learned embeddings.
  Unseen validation categories use a reserved zero-vector entry. This avoids
  expanding high-cardinality IDs into enormous one-hot vectors.
- Model: TabM 0.0.3, eight members, two 128-unit blocks, dropout 0.1.
  This is a bounded initial configuration, not the paper's fully tuned model.
- Optimizer: AdamW, learning rate 0.002, weight decay 0.0003; batch size 1,024.
- Training budget: 20 epochs maximum, early stopping after five nonimproving epochs.
  No label reweighting or resampling. Seed 42 and four CPU threads.
- Train each member's binary cross-entropy independently; at validation, average
  the members' probabilities, not their logits.
- Compute: local Apple M1; use the Apple GPU through PyTorch MPS when available.

## Results

| Model | Validation ROC AUC | Selected training point | Fit wall time |
| --- | ---: | --- | ---: |
| T002-A: CatBoost summaries, CPU | 0.8208754309 | 298 trees | 49.61 s |
| T004: TabM summaries, Apple GPU | 0.8091940040 | Epoch 1 | 27.66 s |

TabM changed AUC by **-0.0116814269** on the exact same sampled rows and holdout.
It stopped after six epochs and restored epoch 1. Preparation took 14.67 seconds.
The timing comparison uses different backends (CPU versus Apple GPU), so it is
not a controlled algorithm-throughput benchmark.

The run showed strong overfitting: training loss fell from 0.19969 at epoch 1
to 0.00484 at epoch 6, while validation AUC fell from 0.80919 to 0.67334.
High-cardinality categorical IDs are a possible contributor, but this trial did
not isolate that cause. This is one small, untuned neural configuration, not
evidence that TabM cannot work on the problem. It does not justify full-data
promotion on its own.

The 656 numerical inputs became 971 channels after adding 315 missingness
indicators. Together with 23 eight-dimensional categorical embeddings, the
backbone received 1,155 numerical channels. No infinite inputs were found.

The selected checkpoint was reloaded and its AUC reproduced; saved validation
probabilities independently reproduce the recorded score. No test predictions,
Kaggle run, or competition submission were generated.

## Reproduction and tracking

```bash
uv sync --extra deep-learning --extra notebooks --locked
uv run --extra deep-learning python tabm_experiment.py \
  --sample-path artifacts/local-v1/sample.parquet --rows 150000 \
  --output-dir artifacts/local-tabm-v1
```

If the original cached sample is unavailable, omit `--sample-path` to regenerate
the seed-42 summary-feature subset from `data/train.parquet`. The notebook freezes
the feature helpers and neural runner used for this trial; later edits to shared
scripts do not change the recorded notebook.

Source revision: `a70a07a9a07a3faad116174536ffa812b951d4b8`.

Outputs are stored in ignored `artifacts/local-tabm-v1/`: best model, fitted
preprocessor, validation probabilities, and results. `results.json`, `run.log`,
and `provenance.json` archive the original metrics, training history, source/data
checksums, and environment versions alongside this README.

Environment: Python 3.12, PyTorch 2.15.0, TabM 0.0.3, NumPy 2.5.4, Polars 1.44.2,
scikit-learn 1.9.1. Apple GPU training can be nondeterministic; the exact original
cached sample is identified by its SHA-256 in the results.

References: official implementation and training guidance in
`yandex-research/tabm`; package version pinned in `pyproject.toml` and `uv.lock`.
