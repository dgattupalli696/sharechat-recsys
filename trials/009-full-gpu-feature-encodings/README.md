# T009 - Full-data scalar and vector encoding ablations

Every model uses all 1,230,110 primary training rows and all 307,528 primary
validation rows. No subset score eliminates any run. Maximum 30 epochs,
five epoch-equivalent patience, full validation every 300 steps plus epoch end.
Keep T008's optimizer/reference settings and seed 42.

| Variant | Only change from the fine-checkpoint reference |
| --- | --- |
| reference | Repeat the unchanged TabM reference |
| ple-16bins-d8 | Train-fitted piecewise-linear scalar embeddings, 16 bins, dimension 8 |
| plr-d8 | Periodic scalar embeddings, dimension/frequencies 8, frequency scale 0.01 |
| quantile-normal | Full-training quantile-normal fit, no row subsampling |
| vector-projection-ln | Per-field 32-D projection, LayerNorm and SiLU; keep magnitude/presence |
| vector-dropout-01 | Drop whole raw-vector fields and their derived summaries with probability 0.1 |
| vector-crosses | Dot/cosine/L2-derived inputs with pair-presence flags |

Encode only original scalar channels with more than 16 distinct training
values. Do not embed binary flags or raw vector coordinates. Retain all other
channels. PLE edges and quantile transforms fit the entire training partition
only. Per-pair dot products use signed log magnitude and L2 uses log distance,
both clipped; cosine stays in [-1, 1]. Missing pairs produce zero interaction
values plus a zero pair-presence flag. All 91 unordered pairs are exploratory:
the anonymous 32-D fields are not known to share a semantic coordinate space.

Reuse the same versioned Kaggle notebook. Frozen sources, original results,
environment, data identities and checkpoint provenance are recorded here.
The custom DCNv2/MoE and FinalMLP-style implementations are also frozen in the
shared feature-model source for subsequent full-data architecture experiments;
those models are not executed in this ablation batch.

No test predictions or competition submissions.

## Completed full-data results

Kaggle notebook version 4 completed all seven variants. Independently
recomputed every AUC and verified labels against T007's full holdout.

| Variant | Validation AUC |
| --- | ---: |
| reference | 0.8292340756 |
| ple-16bins-d8 | 0.8292880922 |
| plr-d8 | 0.8297544465 |
| quantile-normal | 0.8267457317 |
| vector-projection-ln | 0.8296233539 |
| vector-dropout-01 | 0.8290760722 |
| vector-crosses | **0.8353494632** |

Exploratory vector crosses improve AUC by 0.0061153876 over the reference,
without proving semantic alignment of anonymous embedding spaces.
The best checkpoint is at epoch 1; training/prediction took 134.7 seconds.
The next batch tests crosses jointly with high-cardinality masking and
other independently evaluated changes.
