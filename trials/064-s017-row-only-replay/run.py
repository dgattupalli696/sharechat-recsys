"""Frozen S017-style recipes on archived predictions from row-independent models."""
from pathlib import Path
import json
import numpy as np
from sklearn.metrics import roc_auc_score

t=Path(__file__).resolve().parent;a=t.parents[1]/'artifacts';g=a/'kaggle-t062/t062';n=a/'kaggle-t014';ref=np.load(a/'local-t016/validation_predictions.npz');order=ref['row_ids'].argsort();ids=ref['row_ids'][order];y=ref['y_true'].ravel()[order];assert len(np.unique(ids))==len(y)==307528
def load(path):
    with np.load(path) as z:
        order=z['row_ids'].argsort();np.testing.assert_array_equal(ids,z['row_ids'][order]);np.testing.assert_array_equal(y,z['y_true'].ravel()[order]);return z['probabilities'].ravel()[order].astype(float)

k32=load(n/'ensemble-32/validation_predictions.npz');k32seeds=load(a/'local-t018-confirmation-analysis/primary/tabm-k32-plr-cross-masked.npz');fusion=load(n/'cross-masked-fusion-cosine/validation_predictions.npz');depth=load(g/'depth12-holdout.npz');leaf=load(g/'leaves511-holdout.npz');cat=load(g/'catboost-ids-holdout.npz');s022=.75*ref['probabilities'].ravel()[order]+.25*depth
s011=(k32+k32seeds+depth+leaf+cat)/5
rows={name:float(roc_auc_score(y,p)) for name,p in [('S022',s022),('fusion',fusion),('five_family',s011),('six_family',(5*s011+fusion)/6),('S022_plus_fusion',(5*s022+fusion)/6)]};assert abs(rows['S022']-.8415401938202239)<1e-8;print(rows);(t/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
