# T011 - Local full-data LightGBM

Run on the local CPU in parallel with Kaggle neural experiments. Train on all
1,230,110 rows before October 28, 2026, 08:06:52.397 and validate on all
307,528 held-out rows. Validation ordering is explicitly aligned to T008's
original row identities. No subset-based selection or test predictions.

Use the expanded 1,127-feature representation, binary logloss, AUC selection,
learning rate 0.03, 31 leaves, minimum leaf size 100, feature fraction 0.8,
L2 penalty 1, seed 42, maximum 1,000 trees, and 60-round patience.
Categorical vocabularies use training rows only; unseen validation categories
are missing. Save vocabularies, model, aligned probabilities and metrics.

Preprocessing streams the full parquet into temporary float32 memory-mapped
arrays. LightGBM consumes these in batches and stores its binned dataset,
rather than keeping multiple full dense frames in memory. All rows enter
training. Use two CPU threads, column-wise histogram construction and a
512-MB histogram cache; remove owned temporary arrays after completion.
These are local execution changes, not a claim of bitwise reproduction of a
Kaggle/Pandas fit. Native CatBoost embeddings remain on Kaggle.

The notebook and frozen sources reproduce this run from the repository.
Large artifacts stay in `artifacts/local-t011/`, outside Git.
No competition submission.
