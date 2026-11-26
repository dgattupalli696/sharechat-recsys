"""Select the tree share on the earlier fold, then score frozen recipes on primary."""
import json
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

T=Path(__file__).resolve().parent; A=T.parents[1]/'artifacts'; trees=['d12-r0','d12-r1','d6-r0','d6-r1']; weights=[.125,.25,.375,.5,.625,.75,.875,1.]

def read(path):
    with np.load(path) as z:
        ix=z['row_ids'].argsort();return z['row_ids'][ix],z['y_true'].ravel()[ix],z['probabilities'].ravel()[ix].astype('d')

def fold(name):
    ids,y,anchor=read(A/('local-t016/validation_predictions.npz' if name=='primary' else 'local-t018-confirmation-analysis/earlier/frozen-75-25.npz')); pred={}
    for tree in trees:
        i,labels,p=read(A/'kaggle-t063/t063'/f'{name}-{tree}.npz');np.testing.assert_array_equal(ids,i);np.testing.assert_array_equal(y,labels);pred[tree]=p
    pred['all4']=np.mean(list(pred.values()),axis=0);pred['orig12+rich6']=(pred['d12-r0']+pred['d6-r1'])/2
    return ids,y,anchor,pred

def main():
    start=time.monotonic();_,y,a,p=fold('earlier');grid={k:[float(roc_auc_score(y,(1-w)*a+w*v)) for w in weights] for k,v in p.items()};pick=lambda k:(k,weights[int(np.argmax(grid[k]))]); original=pick('d12-r0');winner=pick(max(grid,key=lambda k:max(grid[k])));early_base=grid['d12-r0'][weights.index(.25)];ids,y,a,p=fold('primary');base=.75*a+.25*p['d12-r0'];auc=roc_auc_score(y,base);assert abs(auc-.84154019382)<1e-8; rows={}
    # Request IDs only cluster evaluation uncertainty; never used in scoring.
    import polars as pl
    from importlib.util import spec_from_file_location,module_from_spec
    spec=spec_from_file_location('previous',T.parent/'061-row-only-temporal-confirmation/run.py');mod=module_from_spec(spec);spec.loader.exec_module(mod);rq=pl.read_parquet(T.parents[1]/'data/train.parquet',columns=['request_id'])['request_id'].to_numpy()[ids];groups=np.unique(rq,return_inverse=True)[1]
    for label,(k,w) in [('existing_components',original),('best_overall',winner)]:
        b=(1-w)*a+w*p[k];score=roc_auc_score(y,b);early=grid[k][weights.index(w)];rows[label]=dict(tree=k,tree_weight=w,earlier=early,earlier_gain=early-early_base,primary=score,primary_gain=score-auc,primary_delta_95pct=mod.ci(y,base,b,groups))
    result=dict(weights=weights,earlier_grid=grid,baseline_primary=auc,baseline_earlier=early_base,frozen_results=rows,seconds=time.monotonic()-start);(T/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(rows)

if __name__=='__main__':main()
