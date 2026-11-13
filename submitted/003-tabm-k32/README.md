# S003 - Approved best TabM checkpoint

The user approved this submission and S004 on November 13, 2026:
"do 2 submissions one with highest tabm score and one highest ensemble."

Use the unchanged T014 TabM k32 seed-42 checkpoint, validation AUC
0.8404542697: PLR, exploratory vector crosses and high-cardinality masking.
No refit. Training contains 1,230,110 rows, holdout 307,528 rows.

The shared private inference notebook with S004 uses original model files
and train-fitted preprocessing in the pinned Kaggle training environment.
Verify every bundle checksum and package version, replay the full holdout,
then predict all 1,037,090 test rows and check sample-template IDs.
Export exactly two files; the notebook never submits automatically.

Upload this TabM CSV once after local checksum/schema verification.
Approval is limited to this submission and S004; no other submissions.
