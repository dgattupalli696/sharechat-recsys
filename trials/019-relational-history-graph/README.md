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
