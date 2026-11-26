# T096 – Full-data row-only refit of S025 candidate

Refits our best validated row-only components from T092/T094/T095 on all **1,537,638 labeled rows** and scores all **1,037,090 test rows** strictly row-independently (zero `request_id` usage, singleton invariance verified), then combines them in logit space with the verified S024 full-data predictions (`artifacts/local-t087/submission-full-rowonly-landmarks.csv`). Queued immediately after T095 v58 completes so we can include T095's quantile-PLR neural models if they improve the two-fold validation blend.
