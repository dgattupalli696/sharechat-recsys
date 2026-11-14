# T047 - wide TabM k32, 5-fold on all labeled rows

T027 recipe (8.985 epochs, fixed budget) trained on each 4/5 split of
GroupKFold(5) by request over all 1,537,638 labeled rows (same folds as T046), seeds
42-46. Writes OOF predictions and per-fold plus fold-mean test predictions, to be
averaged with the T037 full refits. Kernel: `sharechat-full-refit-tabm`.
