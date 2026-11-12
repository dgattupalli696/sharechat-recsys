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

Version 8 finished the outer neural fit, but its in-memory LightGBM worker
received SIGKILL. The final tree was fitted locally with disk-backed
matrices, exactly 140 iterations and no early stopping; its predictions
exactly reproduce the independently archived T011 140-tree endpoint.
It used 4.06 GiB peak worker RSS, 71.8 seconds of preparation and 441.0
seconds of fitting/prediction.

## Completed results

| Matched full-outer evaluation | AUC |
| --- | ---: |
| k8 neural, fixed eight epochs | 0.8392367455 |
| LightGBM, fixed 140 trees | 0.8255838643 |
| Equal probability blend | 0.8372661876 |
| Chronological two-logit logistic stack | **0.8394886338** |

The stack improves its matched neural base by 0.0002518883. The conditional
500-replicate request-bootstrap 95% delta interval is
[+0.0000755, +0.0004430]; this is not selection-adjusted or time-block
adjusted. Meta-training used all 943,867 eligible forward OOF rows and no
outer holdout rows.

**T016 remains the overall leader at 0.8408057533.** This two-base prototype
uses a different inference scope and fixed endpoint rather than the
validation-selected k8/k32/DCN checkpoints. Do not interpret that comparison
as isolating stacking itself. No further stacking expansion is scheduled.

The meta-model was fitted locally with scikit-learn 1.9.1. Both failed GPU
attempts, their seven successful base fits, the final CPU fit and the local
assembly are separately recorded in provenance. From the repository root,
`python trials/017-full-temporal-stacking/archive_results.py` verifies all
three source batches and recreates the stack from their saved artifacts.
