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
