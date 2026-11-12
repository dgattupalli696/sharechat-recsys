# T014 - Full-data combinations and architecture comparisons

Every model uses all 1,230,110 primary training rows and all 307,528 primary
validation rows. No subset-based selection or elimination. Maximum 30 epochs
with five epoch-equivalent patience and full validation every 300 updates
plus epoch boundaries. No test inference or competition submission.

First run all eleven recipes in `specs.json`: recheck the vector-cross and
high-cardinality-masked controls, combine those two changes, then separately
test lower LR, two-epoch cosine decay, minimum category count 50, PLR scalar
encoding and per-field vector projection. Also test the PLR/projection fusion
and its lower-LR/cosine alternatives. Do not assume the longer-training masked
recipe benefits from the original two-epoch cosine schedule.

Full-data T009 compared PLE, PLR and quantile-normal alternatives separately;
PLR was strongest. Full-data T009 compared vector projection, dropout and
crosses separately; crosses were strongest and projection was next.
These full-run results motivate the declared combinations, not subset scores.

Select the best of these eleven full-data combinations and run all eight
architecture variants from scratch on that input recipe:

| Variant | Change |
| --- | --- |
| ensemble-16 | TabM k=16 |
| ensemble-32 | TabM k=32 |
| dcnv2-r32 | Two low-rank cross layers, rank 32, plus deep branch |
| dcnv2-r64 | Rank 64 |
| dcnv2-moe4 | Four low-rank cross experts |
| finalmlp-gated | Two gated streams with multi-head bilinear fusion |
| finalmlp-ungated | Disable stream feature gates |
| finalmlp-alternate-lr | Gated model with the other predeclared LR |

All nineteen models receive full-data runs, including k=32 and MoE.
The DCNv2/FinalMLP models are custom paper-inspired implementations, not claims
of reproducing paper-tuned benchmark results. Anonymous vector interactions
are exploratory; no user/item semantics are assumed.

Reuse the canonical versioned Kaggle notebook after T010 outputs are collected.
Archive selected steps, fitted preprocessing, probabilities, hashes and source.

## Completed full-data results

All nineteen models completed in canonical notebook version 6. Saved
probabilities independently reproduce every AUC and match the original full
holdout labels.

| Variant | Validation AUC |
| --- | ---: |
| vector-cross-control | 0.8353494632 |
| masked-category-control | 0.8334188580 |
| cross-masked | 0.8389109395 |
| cross-masked-lr1e4 | 0.8388926127 |
| cross-masked-cosine | 0.8392003042 |
| cross-masked-min50 | 0.8387051189 |
| cross-masked-plr | 0.8400069637 |
| cross-masked-projection | 0.8383806365 |
| cross-masked-fusion | 0.8396135056 |
| cross-masked-fusion-lr1e4 | 0.8396810628 |
| cross-masked-fusion-cosine | 0.8399000037 |
| ensemble-16 | 0.8400283522 |
| ensemble-32 | **0.8404542697** |
| dcnv2-r32 | 0.8389150778 |
| dcnv2-r64 | 0.8384340017 |
| dcnv2-moe4 | 0.8390170406 |
| finalmlp-gated | 0.8388672562 |
| finalmlp-ungated | 0.8386067866 |
| finalmlp-alternate-lr | 0.8392134819 |

The winning input recipe combines vector crosses, high-cardinality masking
and PLR. Its k8/k16/k32 models all selected global step 9,300, or epoch 7.7371.
k32 training/prediction took 1,748.2 seconds; k8 took 618.9 seconds.
The DCNv2/FinalMLP models were individually weaker but remain ensemble inputs.
