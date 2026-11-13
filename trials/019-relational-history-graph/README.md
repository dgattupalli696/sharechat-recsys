# T019 - Relational history and GraphSAGE

Validation-only campaign approved November 13, 2026. No competition submission.

## Frozen design

Audit all 23 anonymous categorical fields using the complete primary training
partition. Select a pair using only the earliest training prefix, before either
holdout, by cardinality, repeated coverage and many-to-many connectivity,
without consulting held-out labels or coverage.
These are entity-like categories, not confirmed user/video identifiers.

The selected pair is `feature5` and `feature18`, with 341,034 and 68,826
training values and 1,230,050 distinct complete training edges. Primary
validation has only 5.8% seen `feature5` values and 91.9% seen `feature18`
values. The severe cold start is a limitation of ID-only collaborative models,
not grounds for choosing a different pair based on holdout performance.

Run both established chronological folds with all available training rows and
all validation rows. Freeze seed 42, batch 1,024 and eight complete epochs,
evaluating the fixed endpoint rather than searching validation checkpoints.

| Model | Purpose |
| --- | --- |
| PLR/cross/masked TabM k8 | Matched no-relational baseline |
| History + same TabM | Past counts, smoothed rates and seen flags for each endpoint and their pair |
| Matrix factorization | Supervised dot-product/bias collaborative baseline on every observed impression |
| Node-only + TabM | Matched graph ablation with node covariates/embeddings but no messages |
| One-layer mean GraphSAGE + TabM | Test additional information from training-only neighbors |

History excludes every label at the current timestamp, including ties.
Validation histories are frozen training-prefix aggregates, never updated
with validation labels. Smoothing uses a fixed 0.05 prior and strength 20.

Graph edges are label-free co-occurrences from training rows only. Deduplicate
edges; sample eight neighbors per node with seed 42. Node covariates contain
training-only means of sixteen standardized continuous inputs, log degree and
endpoint type. The encoder learns both feature transformations and ID
embeddings. Unseen endpoints use current-row covariates and no neighbors;
neither validation nor test rows contribute graph edges or node aggregates.

Use the established pinned Kaggle image and canonical GPU notebook, with each
fold in a fresh subprocess. Archive original validation row identities and all
ten full-model prediction arrays. Compare graph vs node-only, relational models
vs matched control, and primary results vs the existing 0.840806 best ensemble.
The earlier fold is confirmation evidence, not the primary leaderboard proxy.

Version 11 failed at the dependency bootstrap; version 12 passed bootstrap but
its preparation worker received SIGKILL before any model fit. Preserve both
failures. Recovery reuses the original T018 fold-specific, train-fitted
preprocessing (primary byte-identical to T014) and streams every row into
preallocated numeric/category arrays, avoiding full-frame copies and sorting.
Frozen validation row indices preserve the original order. Fold workers launch
directly from the lightweight notebook, not another Torch-importing parent.
The model recipes, complete populations and fixed budgets remain unchanged.

After this campaign is archived, separately refit the selected best model on
all 1,537,638 labeled rows using a validation-frozen training budget. This is
approved training work, not permission for another competition submission.

## Completed results

Canonical Kaggle version 13 completed all ten fits. Executed source, image,
privacy, all original row identities/labels and fixed budgets were independently
verified; primary control exactly reproduces T017's matched neural endpoint.

| Fixed-budget model | Primary AUC | Earlier AUC |
| --- | ---: | ---: |
| TabM control | 0.8392367455 | 0.8355554817 |
| History + TabM | 0.8391971119 | 0.8356107652 |
| Matrix factorization | 0.6918751620 | 0.6836643576 |
| Node-only + TabM | 0.6835815384 | 0.6746812320 |
| GraphSAGE + TabM | 0.7099571127 | 0.6996529262 |

History changes AUC by -0.0000396 primary and +0.0000553 earlier; both
request-clustered conditional confidence intervals cross zero. No useful
improvement is demonstrated. Message passing beats the node-only ablation
by +0.02638/+0.02497, but both variants are much worse than plain TabM.

The collaborative/graph variants severely memorize training entities:
training-probe AUC is 1.0 for matrix factorization and approximately 0.997-0.999
for node/graph models. Primary graph validation log loss is 3.02, compared
with 0.156 for plain TabM. Very low overlap on one endpoint, unregularized
entity embeddings and the untrained unseen-node fallback are limitations of
this tested recipe. These results reject this recipe, not GNNs generally.

The existing T016 four-model ensemble remains the selected approach at primary
AUC 0.8408057533 (already public-scored as S004). Proceed with its separately
approved full-data refit; do not promote the correlation audit into unvalidated
feature changes or submit another CSV without explicit approval.
