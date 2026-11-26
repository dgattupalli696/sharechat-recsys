# T055 – Interval rates and item residual history

Two independent CPU ablations vs saved T054-A (0.89107587056 six-way blend). T036 matrices, original 1,230,110/307,528 chronological split; T054 batched CPU depth12_node, seed 42, six threads, evaluation/early stopping identical. Source baseline revision 6238c1a plus T054 runner. Report global standalone AUC, fixed rank blend, 400 paired request bootstraps and runtime per arm. Promote only at +0.0003 blend AUC for earlier-fold confirmation.

- **rates:** adjacent count differences across f189–192 denominators and f213–216 / f221–224 numerators. Invalid endpoints, nonpositive denominators of differences, or increments outside [0, denominator difference] produce NaN.
- **history:** 4 expanding-window LightGBM teachers, on request-aware scalar predictors, score the next chronological fifth of the fit rows. First fifth is warmup. Fixed 180 rounds, no block labels used to choose the model. For each item, sum strictly earlier out-of-fold `watched − expected_probability`, divided by prior count + 20. No prior gives NaN. Timestamps tied across requests are excluded. Freeze all item histories at the outer cutoff. This feature is tested alone against A, without rate features.

Run `.venv/bin/python -u trials/055-local-rates-item-residuals/run.py` from repository root. Preparation audits, results and confidence intervals are saved here, larger artifacts and run.log in `artifacts/local-t055/`.

## Completed results

| Arm | Standalone AUC | Six-way blend AUC | Blend delta vs T054-A | Paired 95% interval |
| --- | ---: | ---: | ---: | --- |
| Interval rates | 0.889756 | 0.8910754 | −0.0000005 | [−0.0000441, +0.0000481] |
| Item residual history | 0.889706 | 0.8910736 | −0.0000023 | [−0.0000428, +0.0000440] |

Neither variant advances. Item history remains the matched control for the separate T056 directional pair-outcome experiment.
