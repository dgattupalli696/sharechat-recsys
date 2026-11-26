# T059 – Row-independent restart

[Official competition overview](https://www.kaggle.com/competitions/sharechat/overview): prize-eligible solutions must predict each row independently from that row's features. Request context and batch-relative rank blends do not satisfy this condition. Historical request-based public scores do not establish an eligible target.

Build directly from raw numeric inputs and ten same-row user/video embedding cosines. No ID grouping, count, batch ranks or test preprocessing. XGBoost T036 depth12 configuration, CPU / six threads, seed 42, fixed 900 rounds. Fit original chronological prefix, score all 307,528 outer rows, compare to archived T016 using fixed 75% T016 + 25% new probability. Check prediction equality on singletons, reversed and normal batches for the new model. No submission. Promote a positive result of at least +0.0003 for earlier-fold confirmation and full recipe audit. Run `python trials/059-row-only-cosine-xgb/run.py`; results.json records primary metrics and runtime. Large files live in `artifacts/local-t059/`. Baseline revision 6238c1a.
## Result

Standalone 0.8404230; fixed probability blend **0.8415725** vs T016 0.8408058 (+0.0007667). All 307,528 primary holdout rows, 20-row single/batch/reorder invariance probe passed. Proceed to T061 confirmation; no new public score.
