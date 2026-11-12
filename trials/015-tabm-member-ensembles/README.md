# T015 - New TabM and cross-family prediction ensembles

Deferred local prediction-only analysis, automatically starting after T014's
complete outputs/source are downloaded. No new training or Kaggle run.

Use the new full-data PLR + vector-cross + high-cardinality-masked TabM
predictions at k=8, k=16 and k=32. The live logs initially reported
0.8400069637 for k8 and 0.8400283522 for k16; independently recompute final
metrics from saved arrays rather than treating rounded logs as final results.

Compare equal averaging and coarse pair/triple mixtures across those three
models, all three DCNv2 variants, all three FinalMLP variants, T011 LightGBM
and the preceding T013 ensemble. Include all standalone endpoints,
especially k8/k16/k32 alone. Reuse the duplicate-safe ensemble engine
and align every label/row ID on all 307,528 primary holdout rows.

Explicitly test four-family mixtures: one TabM, DCNv2-MoE, one FinalMLP,
and LightGBM. For each of the nine TabM/FinalMLP pairings, evaluate strictly
positive weights on an eighth-step simplex grid, including the 25%-each mean.
Keep pair/triple/standalone candidates so the search does not force weaker
families into the winner. No validation subset or retraining.

Report conditional request-clustered uncertainty against the best single
candidate and the preceding T013 ensemble. These weights are holdout-selected,
not independent temporal confirmation.

Artifacts live in `artifacts/local-t015/`. No competition submission.
