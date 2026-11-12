# T012 - Local LightGBM earlier temporal backtest

Run the same T011 LightGBM recipe on the earlier chronological fold
predeclared before the current campaign:

- Train on every row before October 27, 2026, 08:06:52.397.
- Validate on every row from that cutoff until October 27, 2026, 18:00:00.
- Exclude all later rows from both training and validation.

Use the full available training prefix, not a random subset. Learn categorical
vocabularies and histogram bins from that prefix only. Preserve original
validation row IDs to align with later neural backtests.

The architecture/hyperparameters are unchanged: binary logloss, AUC selection,
maximum 1,000 trees, 60-round patience, 31 leaves, LR 0.03, feature fraction 0.8,
minimum leaf size 100, L2 penalty 1, seed 42, and two CPU threads.
Disk-backed streaming keeps memory bounded while Kaggle neural jobs continue.

This is a temporal robustness backtest, not a retroactively untouched test
set. No primary-holdout result or subset score eliminates an algorithm.
Artifacts live in `artifacts/local-t012/`. No competition submission.

## Completed result

Used all **710,330 training rows and 299,969 validation rows** in this fold.
Validation AUC is **0.8188144776**, selecting 113 trees.
Preparation took 45.3 seconds; training/prediction took 300.4 seconds.
Worker peak resident memory was 4.41 GiB. Keep this score separate from the
primary fold's 0.8255838643: these are different validation populations.
