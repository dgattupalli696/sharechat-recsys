# T080 – joint weighting of eight existing row-only components

Tune on the first chronological half of the earlier fold only, using a differentiable pair-ranking loss on 250k positive/negative pairs (seed 42), simplex constraints and three penalties toward T078. Select interpolation strength or unchanged weights on the later half; freeze for primary. T078's fixed arithmetic/logit pooling and earlier T018 proxy anchor retained. Reused validation predictions; no test ranks or new fits. Promote only for positive gain vs T078 on both and at least +0.0003 primary vs S023. Results in results.json.
