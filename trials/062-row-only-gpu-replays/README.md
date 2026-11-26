# T062 – audit and row-only GPU replay of S004–S017

Canonical private Kaggle GPU notebook version 34 launched Nov 21. This is a training run, not a leaderboard submission.

| Submissions | Disposition |
|---|---|
| S004 | Original row-only probability vote, trained on the primary prefix. Reuse the verified T021 full refit of the same four families; do not retrain it. |
| S005/S006/S007 | Request-context k32 and vote, plus full refit. Stripping the request fields leads back to the pre-request finalist family. Reuse T014/T016/T021 as controls. |
| S008/S009/S010 | Keep XGBoost cosine depth12 and leaves511, test on row-only features. S010's request-wide TabM seeds are not reusable; T018 already tests seed averaging on the row-only k32. Defer leaves63 and more seeds until a distinct benefit is measured. |
| S011/S012 | Recover the CatBoost-with-IDs family using true raw row categorical fields (no request ID), same-row cosines, and only train-fitted CTRs. Validate before full refit. Original S011/S008 test-wide rank blends are invalid; only combine probabilities pointwise. |
| S013/S014 | Defer 5-fold training until a row-only family is selected. If tried, folds based on time or viewer for validation; average model probabilities independently per row. |
| S015/S016 | Weight variants only. No independent model to retrain. Do not transfer weights validated on request features. |
| S017 | Fusion-cosine neural branch worth a **separate** raw-input validation against row-only k32, after current GPU tree batch. Rebuild preprocessing and train from scratch; old checkpoint unusable. No rank blend. |

Fixed T059 depth12 control, 900 rounds CPU→GPU parity check; **full-data depth12 refit on all 1,537,638 labels**, 900 rounds, score 1,037,090 test rows. Validate leaves511 (same rounds) and depth8 CatBoost with non-request categorical IDs. Original chronological holdout, all rows, seed 42. Models only consume raw numeric values, ten same-row embedding cosines and (CatBoost only) the listed raw categorical values. No request ID, test ranks or batch statistics. Includes singleton checks. GPU sources frozen in notebook. Full refit has no independent holdout; T061 earlier-fold evidence is separate. Compare local fixed probability blends to T016 before choosing full refits for additional models.

Postprocess with `python trials/062-row-only-gpu-replays/assemble.py <downloaded-t062-directory>`: validate IDs and full refit count, reconstruct primary holdout blends, generate a full-data pointwise 75% T021 ensemble + 25% GPU XGB CSV. No competition CSV upload or final selection is automated. Base revision 6238c1a plus local T059. Runtime/metrics from worker summaries and assembly report.
