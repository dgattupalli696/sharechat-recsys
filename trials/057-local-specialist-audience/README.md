# T057 – short-request specialist and matched-viewer score

Run `.venv/bin/python -u trials/057-local-specialist-audience/run.py` from the root. T036 wide+cosine matrices and T054-A CPU baseline (0.89107587056 six-way blend); seed 42, six threads, full primary chronological split. Source baseline: revision 6238c1a plus T054/T055 local runners. Stages run sequentially and resume from completed files. Artifacts and log: `artifacts/local-t057/`. No competition submission.

**Specialist**: fit pooled depth12 on first 75% of the training period by row quantile, preserving timestamp ties. Fixed 941 rounds (T054-A endpoint). Same prefix, only requests of size ≤3: fit specialist for 470 fixed rounds. From the remaining 25%, use the earlier half to fit Platt calibrators for singleton and size 2–3 heads, anchored to the long-request pooled scale; on the later half choose the pooled/specialist logit mixing weight from 0, .25, .5, .75, 1 by global pooled-head AUC. No primary holdout rows enter calibration or weight fitting. Refit specialist on all 414,331 short training rows, swap it into the saved T054-A depth12 component only on short rows. Compare global blend vs both size calibration alone and incumbent. Report paired 400-request-bootstrap intervals. Overall ensemble ranks may change as short rows move past long rows.

**Matched viewers**: inside four expanding training prefixes, learn a linear score on differences between positive and negative viewers shown the same item, matched on request size group (1, 2–3, 4–6, 7+); different `feature5` and request IDs required. Up to 3 sampled negatives per positive; exact pairs deduplicated. Use audience scalars, cosines, request ranks and audience×duration products. Fit transforms and L2 logistic classifier on prefix only, then score the next fifth. Warmup scores missing. For primary holdout use the classifier fitted on the entire training partition. Use its score as the sole new input to T054-A's XGBoost recipe. Support check, full training prefix with seed 42: 149,622 distinct sampled pairs / 56,308 watched rows / 12,580 distinct videos. No pair generation on outer holdout labels.

**Decision**: +0.0003 global blend AUC vs T054-A for earlier-fold confirmation; the specialist also needs a positive delta over recalibration alone to attribute a gain to specialization. No refit or submission scheduled.

## Completed results

| Variant | Global blend AUC | Delta vs T054-A (paired 95% interval) |
| --- | ---: | --- |
| Size calibration only | 0.8910744 | −0.0000014 [−0.0000053, +0.0000027] |
| Specialist (25% inner-selected weight) | 0.8910737 | −0.0000022 [−0.0000125, +0.0000081] |
| Matched-viewer score | 0.8910571 | −0.0000188 [−0.0000607, +0.0000326] |

The specialist also trailed recalibration alone by 0.0000007. The audience model ordered 70.5–71.2% of held-out pairs correctly across four forward blocks, but the downstream fit (standalone AUC 0.8897081 at round 875) did not improve the ensemble. Neither advances.
