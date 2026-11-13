# T024 - Request context in the selected full models

T023 showed label-free within-request context lifts a scalar LightGBM screen
from 0.8287 to 0.8826 on the original holdout. T024 adds the same 33 request
features (`request_features.py`: request size, distinct items, duplicate-item
count, and within-request rank/deviation/gap-to-max of the ten most
target-correlated numeric features) to the full selected models, using the
original T014/T011 protocols unchanged:

- Kaggle T4: TabM k8 (`cross-masked-plr`), TabM k32 (`ensemble-32`) and
  DCNv2-r32, original primary holdout, 30-epoch cap, 300-step validation,
  patience 5.
- Local CPU: full expanded LightGBM (T011 protocol: lr 0.03, 31 leaves,
  1,000-tree cap, 60-round patience), aligned to T011 holdout identities.

Compare against T014/T011 and the T016 ensemble (0.840806). Selection uses the
original holdout. No test predictions or competition submission.

## Results (Kaggle v21 + local LightGBM)

Same 307,528-row holdout as T014 (label SHA `990ee36e…`); predictions aligned
by row id. Executed frozen sources match the repository.

| Model | T014/T011 AUC | T024 AUC | Best epoch / trees |
|---|---:|---:|---:|
| TabM k32 (`ensemble-32`) | 0.840454 | **0.889350** | 17.73 |
| TabM k8 (`cross-masked-plr`) | 0.840007 | **0.888989** | 8.99 |
| DCNv2-r32 | 0.838915 | **0.887607** | 2.50 |
| LightGBM (full, expanded) | 0.825584 | **0.880290** | 255 trees |
| Ensemble, T016 weights (.45/.225/.225/.10) | 0.840806 | **0.889809** | |
| Ensemble, 0.05-grid best (.50/.25/.15/.10) | | 0.889825 | |

Rank-average ensembling on the holdout; the grid gain over fixed T016 weights
is negligible, so T016 weights are kept. Component rank correlations are
0.986-0.992 among neural models and 0.944-0.957 with LightGBM.

Kaggle v16-v20 were OOM-killed (32 GB) during data preparation; fixed without
changing prepared arrays (bitwise-verified): blockwise float32 conversion,
no redundant rechunk, column-wise sort, and building the sorted frame in a
child process then memory-mapping it (`SHARECHAT_SAMPLE_CACHE`).
Files: `neural-results.json`, `lightgbm-results.json`, `ensemble.json`,
`protocol.json`, `run.log`, `local-lightgbm.log`.
