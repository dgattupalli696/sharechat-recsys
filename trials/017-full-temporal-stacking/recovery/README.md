# T017 recovery - isolated full outer base fits

Canonical notebook v7 completed all six fixed-budget base fits for the
three forward OOF periods, then its Python kernel died during preparation
of the full outer fold. No outer result or stack score was produced.
The last message was `DeadKernelError`; accumulated memory is a plausible
cause, not a confirmed platform diagnosis.

Preserve the original snapshot, logs and six verified base fits. Run only
the missing full outer neural and LightGBM fits in this recovery notebook,
each in a fresh Python process so its allocator memory is released at exit.
The temporary worker modules are exact frozen sources, not modifications
to tracked files. Use the same input recipe, seed, eight neural epochs,
140 tree iterations, complete training partition and complete holdout.

After download, combine the original three OOF blocks with the recovered
outer fits locally. Fit the original two-logit logistic meta-model only
on the 943,867 OOF rows; the initial 286,243 warmup training rows are
excluded. Record the local meta-model library version and both source
notebook versions. Do not retrain already completed bases, use subsets
or submit to the competition.
