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

## Completed full-holdout results

Evaluated 1,158 configurations from eleven distinct prediction arrays.

| Configuration | Validation AUC |
| --- | ---: |
| Best single: TabM k32 | 0.8404542697 |
| Equal k8 + k16 | 0.8401713074 |
| Equal k8 + k32 + DCNv2-r32 | 0.8406773179 |
| 62.5% k32 + 12.5% each MoE / gated FinalMLP / LightGBM | 0.8406704426 |
| 50% k32 + 25% k8 + 25% DCNv2-r32 | **0.8407075087** |

Adding a weaker DCNv2 model helped: the best three-model mixture gains
0.0002532390 over k32 alone. Its 500-replicate request-clustered conditional
95% delta interval is [+0.0000966, +0.0004165]. This does not account for
holdout model/weight selection and is not temporal confirmation.
The four-family mixture also beat the best single, but not the simpler winner.
