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

## Completed results

Full-data LightGBM achieved **0.8255838643 AUC**, selecting 140 trees.
Preparation took 70.0 seconds; training and prediction took 563.6 seconds.
The isolated worker's measured peak resident memory was 5.03 GiB.
All holdout labels and original row identities exactly match T008.

A preliminary probability blend of **75% T008 high-cardinality-masked TabM
and 25% LightGBM** achieved **0.8343357592 AUC**, versus TabM's
0.8334188580: an improvement of 0.0009169013.
The coarse grid tested tree weights 0%, 25%, 50%, 75%, and 100%.

A 500-replicate paired bootstrap over 85,985 request groups produced a
conditional 95% delta interval of [+0.000614, +0.001244]. This interval is
conditional on validation-selected models/weights, not selection-adjusted
or time-block-adjusted confirmation. Later models and robustness folds remain
outstanding; this is not a final ensemble or a competition submission.
The separate blend notebook freezes the exact post-hoc analysis.
