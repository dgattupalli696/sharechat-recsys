# T013 - Existing full-data model ensembles

Compare existing predictions on the same complete 307,528-row primary holdout.
No retraining or bootstrap resampling of training rows: this is prediction
averaging/blending, rather than classical bagging.

Inputs are all 12 T008 controls, all seven T009 encodings, the T006 attention
network, both T007 neural variants, and T011 LightGBM. Exclude subset-trained
models and the earlier temporal fold. Deduplicate identical prediction arrays
before assigning ensemble weights.

Evaluate equal means of all unique models, the top two through eight models,
each model family, family-balanced averaging, and every model pair at
25/75, 50/50 and 75/25 weights. For an explicitly recorded eight-model core,
also evaluate all three-model equal means and 50/25/25 permutations.
Keep the best single-model endpoint as a candidate.

The core spans vector-cross TabM, high-cardinality-masked TabM, cosine TabM,
periodic TabM, projected-vector TabM, attention, attention/TabM and LightGBM.
Weights and membership are chosen on the same holdout: results are exploratory,
not independent confirmation. Report request-clustered paired uncertainty
against the best current single model and the previous two-model blend.

Original legacy files did not store row IDs. The input preparation checks
their archived file hashes, full row counts, cutoff, identical pinned image,
common ordered-loader source, byte-identical fitted preprocessing and matching
labels before deriving IDs from T008. T007 standalone additionally matches
T008's epoch control predictions exactly. Preserve these derivation
certificates rather than claiming the IDs came from original legacy outputs.

Run the frozen notebook from the repository. Large predictions remain in
`artifacts/local-t013/`. No competition submission.
