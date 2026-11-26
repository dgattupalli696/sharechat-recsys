# T056 – cross-request item pairs

Primary chronological holdout and full T054-A CPU protocol, six fixed-rank-blend members, seed 42. Run `python trials/056-local-pair-history/run.py` after T055 has produced both fit results. Code baseline: T055 and T054 local runners, original T036 inputs (base revision 6238c1a). Run log and bulky prediction artifacts in `artifacts/local-t056/`.

1. Audit coverage at **candidate row** level. Count distinct requests per pair; discard same-time evidence, freeze counts at cutoff. No holdout labels contribute. Reproduce the forward T055 teacher and check item-history parity against its archived features.
2. Fit popularity control (log prior item exposure and maximum companion exposure). Then add the fraction of previously seen companions and strongest previously observed pair lift `n/(n+5) * log((n+1)/(expected+1))`, where `expected=item_count*companion_count/earlier_requests`. Compare association vs popularity as well as T054-A.
3. Directional outcome: among past A+B requests, aggregate A's residual after its T055 expected watch probability **and** earlier item effect. Require 3 earlier OOF outcomes per qualifying partner, pool their sums with 20 pseudo-observations per partner. Missing when none qualifies. Compare item-history + pair effect against saved T055 item-history alone. Skip this fit unless at least 10% of candidate rows have support in both fit and holdout; criterion uses availability, not holdout outcomes.

Primary outcome: global six-way blend delta; 400 paired request bootstraps. Require positive incremental delta over the matched control and at least +0.0003 blend AUC vs T054-A for earlier-fold confirmation.

## Completed results

On the 307,528 validation rows, 54.95% had a previously seen companion and 29.64% qualified for directional outcomes (fit: 54.04% / 21.37%). All three fits ran. Popularity-only blend: 0.8910630; T055 item-history blend: 0.8910736; T054-A: 0.8910759.

| Variant | Blend AUC | Delta vs matched control (paired 95% interval) | Delta vs T054-A |
| --- | ---: | --- | ---: |
| Association | 0.8910753 | +0.0000124 vs popularity [−0.0000323, +0.0000599] | −0.0000006 |
| Directional pair outcome | 0.8910676 | −0.0000059 vs item history [−0.0000497, +0.0000379] | −0.0000082 |

Neither reaches the promotion threshold. Retain the incumbent; no refit or submission.
