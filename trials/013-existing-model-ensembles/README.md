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

## Completed full-holdout results

Evaluated 883 configurations from 23 inputs / 21 distinct prediction arrays.
Removed the duplicate T009 reference and T007 standalone copies.

| Ensemble | Validation AUC |
| --- | ---: |
| Best single: vector-cross TabM | 0.8353494632 |
| Equal average of all 21 distinct models | 0.8317015472 |
| Family-balanced average | 0.8317370591 |
| Best equal mean: masked-category TabM + vector-cross TabM | **0.8366686874** |
| 25% masked-category TabM + 50% vector-cross TabM + 25% LightGBM | **0.8368991177** |

The three-model blend gains 0.0015496544 over the best current single model
and 0.0025633585 over the preceding two-model blend. A 500-replicate paired
request bootstrap gives a conditional 95% delta interval of
[+0.000995, +0.002088] against the best single. Membership and weights were
selected on this holdout; this is not selection-adjusted confirmation.

Blindly adding models reduced AUC. The bounded search prefers complementary
strong neural recipes plus a smaller tree contribution, not the largest bag.
