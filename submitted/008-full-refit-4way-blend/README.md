# S008 — Full-data 4-way rank blend (TabM k32 wide + 3 XGBoost)

User approval: "lets do 2 submission of the 4 way blend with hold out and full data"

- Components, all refit on 1,537,638 labeled rows: T037 TabM k32 wide (seed mean of 42/43/44) and T038 XGBoost + cosines (leaves63, leaves511, depth12; seed mean of 42/43).
- Rank blend 0.45 / 0.15 / 0.15 / 0.25, weights chosen on the original holdout with the train-only T027/T036 models (holdout AUC 0.89081).
- Submission 58332687: **public 0.88641**, +0.00174 vs S007 (0.88467). Best public score.
