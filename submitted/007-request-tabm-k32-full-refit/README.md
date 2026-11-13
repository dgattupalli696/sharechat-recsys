# S007 — Request-context TabM k32, full-data refit

User approval: "submit t025 now lets see if it helps at all"

- Model: T025 — T024 TabM k32 recipe (33 request-context features) retrained on all 1,537,638 labeled rows for a fixed 17.73 epochs (26,629 steps); no holdout score by design.
- Kaggle kernel v23, T4; training 2,519 s. See [T025](../../trials/025-request-tabm-k32-full-refit/).
- Submission 58304230: **public 0.88467** — +0.00073 vs S005 (same recipe, train split only, 0.88394) and +0.00029 vs S006 (ensemble, 0.88438). Best public score.
- Conclusion: full-data refit helps with request features; refitting the full ensemble is the next logical step.
