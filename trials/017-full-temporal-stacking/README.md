# T017 - Full-prefix chronological OOF stacking

Test a two-base logistic stack on every available row of three forward
out-of-fold periods inside the original training partition. Keep all
307,528 primary holdout rows outside meta-model fitting. No random subsets,
test inference or competition submission.

Use the full-data PLR + vector-cross + high-cardinality-masked TabM k8
recipe and expanded-feature LightGBM. k8 is the efficient, fully evaluated
version of the best input recipe (0.840007 AUC), rather than a subset proxy.
The best k32 single scored 0.840454; report these as different inference scopes.
The smaller backbone keeps repeated full-prefix fitting practical.

Freeze neural training at eight epochs (ceiling of the independently selected
7.7371-epoch checkpoint) and trees at 140 iterations (the full-data LightGBM
selection). Do not use any OOF-block labels for base early stopping or
checkpoint selection. Fit all preprocessing on each training prefix only.

| Training prefix ends before | OOF prediction period ends before |
| --- | --- |
| October 26, 2026, 12:00 | October 27, 2026, 00:00 |
| October 27, 2026, 00:00 | October 27, 2026, 12:00 |
| October 27, 2026, 12:00 | October 28, 2026, 08:06:52.397 |

Exclude initial warmup rows with no forward OOF prediction. Check that OOF
row IDs are disjoint and every base has identical targets/row order.
Fit LogisticRegression(C=1, max_iter=1000) to clipped base logits.
Fit matched fixed-budget base models on the full primary training partition,
then predict the outer holdout once for stack evaluation.

Report the stack against the matched fixed-budget neural/tree models and
their equal probability blend, not only against differently early-stopped
models. Hyperparameters were already selected using the primary holdout:
this experiment is not a retroactively untouched evaluation.

Reuse canonical Kaggle notebook versioning. The existing conditional positive
gain from cross-model blending motivates this full OOF experiment. Earlier
time-fold and multi-seed confirmation remain separate work.

## Recovery

Version 7 completed all three OOF periods (943,867 validation predictions
per base in total), then the kernel died during outer-fold preparation.
All six base fits were preserved. The [recovery notebook](recovery/README.md)
isolates the two remaining full outer fits into fresh processes. No OOF
block is reduced, discarded or selected based on its AUC.
