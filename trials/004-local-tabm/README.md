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

Pending local execution. Reference T002-A validation AUC: 0.8208754309.
The best checkpoint is reloaded and its validation AUC checked before results
are finalized. No test predictions or competition submission are generated.

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

Outputs are stored in ignored `artifacts/local-tabm-v1/`: best model, fitted
preprocessor, validation probabilities, and results. Small metrics, logs, source
and data checksums, and environment versions will be archived beside this README.
This trial does not run Kaggle compute or submit anything.

References: official implementation and training guidance in
`yandex-research/tabm`; package version pinned in `pyproject.toml` and `uv.lock`.
