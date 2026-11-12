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
