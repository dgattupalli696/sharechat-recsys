# T020 - Local full-training feature correlations

Completed November 13, 2026 in 85.1 seconds on the local M1, limited to four
BLAS threads. No model fits, row subsampling, leaderboard submissions or changes
to the frozen T019 experiment.

## Data boundaries

Numeric and categorical redundancy use all 1,230,110 primary training rows.
Numeric target associations use only the earliest 710,330-row training prefix,
before either outer holdout. Categorical target rates use an additional
chronological 80/20 split inside that prefix, with a train-fitted prior and
smoothing strength 20. Neither outer holdout's labels inform this audit.

All 1,104 engineered numeric columns are audited, including the 32 coordinates
and summaries of every vector. Correlations use pairwise finite observations,
not mean imputation or complete-case row deletion. Standardized float32 block
products are accumulated in float64; the matrix is a screening tool, with
finite-observation counts included. Exact duplicates use independent hashes.

## Findings

| Finding | Evidence |
| --- | --- |
| Strongest numeric target associations | `feature418`: r=0.1571; `feature419`: 0.1562; `feature411`: 0.1453; `feature410`: 0.1449 |
| Exact numeric duplicate | `feature3` equals engineered timestamp hour on every primary training row |
| Strong numeric redundancy | 330 pairs have absolute Pearson correlation at least 0.98 |
| Constant when observed | All 14 vector-length columns are 32 whenever present; their missingness may still be informative |
| Equivalent categorical partition | `feature18` and `feature24` determine each other with zero exceptions, including missing values |
| Strongest categorical rate signal | Either `feature18` or `feature24` alone gives inner forward AUC 0.702476 |
| Other categorical rate signals | `feature20`: inner AUC 0.670207; `feature23`: 0.618011 |

These are associations, not causal findings or multivariate feature importance.
The inner categorical AUCs are not comparable to the primary outer-holdout AUCs.
High numeric correlation does not establish that dropping a field improves a
nonlinear model. Do not remove the constant vector lengths without preserving
their missingness indicators.

The entity-rate finding supports the already-frozen T019 history experiment:
the selected TabM recipe masks high-cardinality IDs, whereas history rates can
provide a regularized, lower-dimensional signal. It does not establish that
the feature will improve either outer holdout or explain the leaderboard gap.
No feature-selection changes have been applied yet.

## Artifacts and reproduction

`results.json` contains complete summaries, strong pairs and categorical
dependencies. The CSVs list every numeric feature and categorical rate result.
The full correlation/count matrices are in ignored
`artifacts/local-t020-correlations/numeric-correlations.npz`; their SHA-256 and
the exact frozen source hashes are in `provenance.json`.

The notebook executes the frozen local audit code against the original dataset.
