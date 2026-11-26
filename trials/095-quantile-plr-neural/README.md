# T095 – Train-only quantile-encoded Enriched TabM k32 + DCNv2-MoE4 (2 seeds each)

Combine T081's train-only 33-knot quantile percentile encoding with T090/T094's strong PLR + pairwise vector-cross + low-cardinality categorical neural architectures (`QuantileTabM` width 256 k32, and `DCNv2-MoE4` width 256) on the 1,240 super-enriched row-only features, 2 seeds each (42, 43) across both chronological folds. Strictly row-independent; zero `request_id` usage; singleton invariance verified. Joint Caruana logit ensemble evaluated against S024. No submission.
