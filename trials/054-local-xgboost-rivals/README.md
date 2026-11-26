# T054 – Local CPU rival-feature ablation

Source baseline: revision `6238c1a`, T036 `depth12_node`. Run from repository root with `.venv/bin/python trials/054-local-xgboost-rivals/run.py`. The notebook invokes the same driver. All three arms use seed 42, six CPU threads, max_bin 127, identical batch construction, and the original chronological split (1,230,110 fit, 307,528 validation). Use T036 holdout early stopping (300 rounds); report global AUC and a six-way rank blend replacing only depth12. Freeze the other five predictions and weights. Gate: C beats B and gains at least 0.0003 blend AUC vs A before earlier-fold confirmation.

A: cached T036 predictors. B: A plus count of distinct other items sharing feature23 in the request. C: B plus affinity minus best same-creator affinity, affinity minus nearest item's affinity, and duration minus nearest item's duration. Affinity is cos_32_43; nearest is by feature43 cosine, ties broken by lexical item ID then file order. Ignore missing/zero vectors for similarity and affinity; absent comparisons are NaN. No labels enter these four features.

Trial artifacts, checkpoints (completed arms), and progress log: `artifacts/local-t054/`. Small result summaries are written here after each fit. Runtime and peak process RSS are recorded per arm (macOS). No test inference or submission.

## Completed A/B/C results

| Arm | Standalone AUC | Six-way blend AUC | Blend delta vs A |
| --- | ---: | ---: | ---: |
| A baseline | 0.889754 | 0.891076 | — |
| B count | 0.889843 | 0.891104 | +0.000028 |
| C count + gaps | 0.889864 | 0.891091 | +0.000015 |

C−B blend delta −0.000013; all three paired bootstrap intervals span zero (`bootstrap.json`). Neither variant meets the promotion gate.

## D – learned candidate match (independent of B/C)

After the A/B/C driver exits, run `.venv/bin/python trials/054-local-xgboost-rivals/match.py`. Three request-hash folds; train each match classifier on observed pre-cutoff pairs vs other-request items matched by absolute hour, duration bin, and log2 fit-frequency bin. Exclude every item already present in the recipient request. Use 160k paired examples per fold (or all if fewer), fixed L2 logistic regression on 96 coordinate products from three user/item embedding pairs. No watch outcomes or IDs are model inputs. Out-of-fold match logit and within-request percentile rank are the only additions to A. Score each outer validation request with its hash-assigned fold model, as for training rows. Same downstream parameters, holdout and blend evaluation as A; 400 paired request bootstraps for D−A. Primary gate: +0.0003 blend AUC vs A before earlier-fold confirmation.

Initial 1/32 request-hash smoke check: unseen-request real-vs-swapped AUC 0.73427 / 0.74077 / 0.72718 across folds. This measures the auxiliary task only. Full run log: `artifacts/local-t054/match-run.log`.

Full run: auxiliary held-out pair AUC 0.72683 / 0.72715 / 0.72708. Watch AUC 0.889794 standalone (+0.000040 vs A), 0.891074 blended (−0.000002 vs A; paired request-bootstrap 95% interval [−0.000044, +0.000044]). No earlier-fold confirmation or refit warranted. Best round 814, CPU fit 41.7 minutes, peak 7.33 GiB.
