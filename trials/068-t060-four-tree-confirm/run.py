"""Fixed-weight earlier-fold T060 add-on; reuse T061 fold and CPU T059 comparator."""
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import roc_auc_score

T=Path(__file__).resolve().parent;ROOT=T.parents[1];A=ROOT/'artifacts';OUT=A/'local-t068'
spec=importlib.util.spec_from_file_location('t061',T.parent/'061-row-only-temporal-confirmation/run.py');prev=importlib.util.module_from_spec(spec);spec.loader.exec_module(prev);pr=prev.priors

def score(fold,ids,y,p,q,groups):
    def aligned(path):
        with np.load(path) as z:
            ix=z['row_ids'].argsort();np.testing.assert_array_equal(ids,z['row_ids'][ix]);np.testing.assert_array_equal(y,z['y_true'].ravel()[ix]);return z['probabilities'].ravel()[ix].astype('d')
    anchor=aligned(A/('local-t016/validation_predictions.npz' if fold=='primary' else 'local-t018-confirmation-analysis/earlier/frozen-75-25.npz'));b=.5*anchor+.125*sum(aligned(A/f'kaggle-t063/t063/{fold}-{k}.npz') for k in ['d12-r0','d12-r1','d6-r0','d6-r1']);aug=(5*b+p)/6;ctrl=(5*b+q)/6;base=roc_auc_score(y,b);assert abs(base-({'primary':.84198952,'earlier':.83851422}[fold]))<1e-7
    return dict(rows=len(y),base_auc=base,plus_T060_auc=roc_auc_score(y,aug),gain=roc_auc_score(y,aug)-base,gain_95pct=prev.ci(y,b,aug,groups),plain_tree_gain=roc_auc_score(y,ctrl)-base,priors_vs_plain=roc_auc_score(y,aug)-roc_auc_score(y,ctrl),priors_vs_plain_95pct=prev.ci(y,ctrl,aug,groups))

def main():
    OUT.mkdir(exist_ok=True,parents=True);start=time.monotonic();f=pl.read_parquet(ROOT/'data/train.parquet',columns=['feature1','watched',*pr.COLS]);tr,va,pre=prev.masks(f);yy=f['watched'].to_numpy();rq=pl.read_parquet(ROOT/'data/train.parquet',columns=['request_id'])['request_id'];groups=lambda m:np.unique(rq.filter(pl.Series(m)).to_numpy(),return_inverse=True)[1]
    def saved(path,ids,key):
        with np.load(path) as z:np.testing.assert_array_equal(z['row_ids'],ids);return z[key].astype('d')
    ids=np.flatnonzero(~pre);primary=score('primary',ids,yy[~pre],saved(A/'local-t060/predictions.npz',ids,'probabilities'),saved(A/'local-t059/predictions.npz',ids,'probabilities'),groups(~pre));(T/'primary.json').write_text(json.dumps(primary,indent=2)+'\n');print(primary,flush=True)
    ids=np.flatnonzero(va);q=saved(A/'local-t061/059.npz',ids,'predictions');h=f.filter(pl.Series(tr));ts=h['feature1'].cast(pl.Int64).to_numpy();blocks=np.searchsorted(np.quantile(ts,[.1,.2,.4,.6,.8],method='inverted_cdf'),ts,side='right');fx=np.full((len(h),2*len(pr.KEYS)),np.nan,'f4')
    for k in range(1,6):
        train=blocks<k;test=blocks==k;assert ts[train].max()<ts[test].min();fx[test]=pr.apply(h.filter(pl.Series(test)).select(pr.COLS),pr.learn(h.filter(pl.Series(train))))
    extra=[fx,pr.apply(f.filter(pl.Series(va)).select(pr.COLS),pr.learn(h))];x=np.load(A/'local-t059/fit.npy',mmap_mode='r');indices=[np.flatnonzero(m[pre]) for m in (tr,va)];labels=[yy[tr],yy[va]];np.testing.assert_array_equal([len(i) for i in indices],[710330,299969]);np.testing.assert_array_equal(yy[pre],np.load(A/'local-t060/y_fit.npy'))
    class Batch(xgb.DataIter):
        def __init__(self,i):self.i=i;self.reset();super().__init__()
        def reset(self):self.at=0
        def next(self,input_data):
            if self.at>=len(indices[self.i]):return False
            s=slice(self.at,self.at+8192);self.at+=8192;input_data(data=np.column_stack([x[indices[self.i][s]],extra[self.i][s]]),label=labels[self.i][s]);return True
    train=xgb.QuantileDMatrix(Batch(0),max_bin=127,nthread=6);valid=xgb.QuantileDMatrix(Batch(1),ref=train,max_bin=127,nthread=6);model=xgb.train({**prev.BASE,**prev.CONFIGS['depth12_node'],'device':'cpu','nthread':6},train,900,evals=[(valid,'earlier')],verbose_eval=200);p=model.predict(valid);np.savez_compressed(OUT/'earlier.npz',row_ids=ids,probabilities=p);earlier=score('earlier',ids,yy[va],p,q,groups(va));result=dict(primary=primary,earlier=earlier,seconds=time.monotonic()-start);(T/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)

if __name__=='__main__':main()
