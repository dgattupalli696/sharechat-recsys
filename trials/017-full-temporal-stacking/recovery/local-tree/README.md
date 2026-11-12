# Final full-data LightGBM fit on local CPU

Version 8 completed the full fixed-eight-epoch neural fit (0.8392367455
AUC), then the in-memory LightGBM worker received SIGKILL. Preserve that
neural checkpoint and all six earlier OOF fits; do not rerun them.

Use the established streaming float32 feature matrices and isolated,
disk-backed LightGBM worker locally. All engineered numeric features are
already float32 in the original path; categorical vocabularies retain
training-time appearance order in both implementations. A focused
comparison checks identical predictions against the original in-memory
fixed-budget implementation. Use the same predictive settings, seed 42,
all 1,230,110 training rows and all 307,528 holdout rows.

Train exactly 140 iterations with no validation-driven early stopping.
The only execution changes are memory mapping, explicit column-wise
histograms, a 512-MB histogram cache and two local CPU threads rather than
four cloud threads. These resource settings are recorded; do not claim
bitwise agreement across different hosts or library builds.

Align validation rows to the saved outer neural identities before fitting.
After completion, use the three original OOF blocks, recovered outer neural
and this outer tree to evaluate the original two-logit logistic stack.
The primary holdout remains outside meta-model fitting. No submission.
