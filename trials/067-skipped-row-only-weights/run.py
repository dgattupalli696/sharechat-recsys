"""Revisit skipped probability components; select on earlier half, evaluate later."""
import json
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

t=Path(__file__).resolve().parent;a=t.parents[1]/'artifacts';ref=np.load(a/'local-t016/validation_predictions.npz');ix=ref['row_ids'].argsort();ids=ref['row_ids'][ix];y=ref['y_true'].ravel()[ix];anchor=ref['probabilities'].ravel()[ix].astype('d')
def load(path):
    with np.load(a/path) as z:
        order=z['row_ids'].argsort();np.testing.assert_array_equal(ids,z['row_ids'][order]);
        if 'y_true' in z:np.testing.assert_array_equal(y,z['y_true'].ravel()[order])
        return z['probabilities'].ravel()[order].astype('d')
def nn(k):return load(f'kaggle-t014/{k}/validation_predictions.npz')
def tree(k):return load(f'kaggle-t062/t062/{k}-holdout.npz')
depth=tree('depth12');leaf=tree('leaves511');cat=tree('catboost-ids');f=nn('cross-masked-fusion-cosine');five=(nn('ensemble-32')+load('local-t018-confirmation-analysis/primary/tabm-k32-plr-cross-masked.npz')+depth+leaf+cat)/5
base={'S022':.75*anchor+.25*depth,'T065':.5*anchor+.125*sum(load(f'kaggle-t063/t063/primary-{k}.npz') for k in ['d12-r0','d12-r1','d6-r0','d6-r1'])};candidates={'fusion':f,'six_slot':(5*five+f)/6,'leaves511':leaf,'catboost_ids':cat,'T060_priors':load('local-t060/predictions.npz')};
ts=pl.read_parquet(t.parents[1]/'data/train.parquet',columns=['feature1'])['feature1'].cast(pl.Int64).to_numpy()[ids];cut=np.median(ts);first=ts<cut;last=~first;ws=[0,.025,.05,.1,1/6,.25,.375,.5,.75,1];results={}
for name,b in base.items():
    scores={};baseline=[roc_auc_score(y[m],b[m]) for m in (first,last)];assert abs(roc_auc_score(y,b)-{'S022':.84154019382,'T065':.84198952111}[name])<1e-8
    for key,p in candidates.items():
        vals=[roc_auc_score(y[first],((1-w)*b+w*p)[first]) for w in ws];j=int(np.argmax(vals));w=ws[j];d=roc_auc_score(y[last],((1-w)*b+w*p)[last])-baseline[1];scores[key]=dict(weight=w,selection_gain=vals[j]-baseline[0],later_gain=d)
    results[name]=scores
report={'rows':[int(first.sum()),int(last.sum())],'weights':ws,'results':results};(t/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
