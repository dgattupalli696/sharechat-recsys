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
