"""Add frozen, train-only categorical counts and watch priors to row-only T059."""
import json
from pathlib import Path
import sys
import time

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
import xgboost as xgb

TRIAL=Path(__file__).resolve().parent;ROOT=TRIAL.parents[1];sys.path.insert(0,str(ROOT))
from feature_screen import CUTOFF
from xgb_tune import BASE,CONFIGS

# Anonymous viewer-profile and content fields, not request identifiers.
KEYS=[(k,) for k in (6,7,9,12,14,18,19,20,23)]+[(6,19),(7,19),(9,20),(12,23)]
COLS=[f'feature{i}' for i in sorted({i for key in KEYS for i in key})];OUT=ROOT/'artifacts/local-t060';M=ROOT/'artifacts/local-t059'


def learn(frame):
    maps=[]
    for key in KEYS:
        cols=[f'feature{i}' for i in key]; tab=frame.drop_nulls(cols).group_by(cols).agg(pl.len().alias('n'),pl.col('watched').cast(pl.Int64).sum().alias('s'));maps.append((cols,tab))
    return float(frame['watched'].mean()), maps


def apply(frame,state):
    prior,maps=state; parts=[]
    for cols,tab in maps:
        joined=frame.select(cols).join(tab,on=cols,how='left',maintain_order='left',validate='m:1'); n=joined['n'].fill_null(0).to_numpy(); s=joined['s'].fill_null(0).to_numpy(); parts.extend([np.log1p(n),(s+40*prior)/(n+40)])
    return np.column_stack(parts).astype('f4')


def prepare():
    f=pl.read_parquet(ROOT/'data/train.parquet',columns=['feature1','watched',*COLS]);v=(f['feature1']>=CUTOFF).to_numpy();ids=np.flatnonzero(v);hist=f.filter(pl.Series(~v));evalrows=f.filter(pl.Series(v));t=hist['feature1'].cast(pl.Int64).to_numpy();edges=np.quantile(t,[.1,.2,.4,.6,.8],method='inverted_cdf');blocks=np.searchsorted(edges,t,side='right');extra=np.full((len(t),2*len(KEYS)),np.nan,'f4')
    for k in range(1,6):
        tr=blocks<k;va=blocks==k;assert t[tr].max()<t[va].min();extra[va]=apply(hist.filter(pl.Series(va)).select(COLS),learn(hist.filter(pl.Series(tr))))
    fixed=learn(hist);vx=apply(evalrows.select(COLS),fixed);sample=[2,17,12,3,0,11];np.testing.assert_array_equal(vx[sample],apply(evalrows[sample].select(COLS),fixed));np.testing.assert_array_equal(vx[2:3],apply(evalrows.slice(2,1).select(COLS),fixed));assert np.isfinite(extra[blocks>0]).all() and np.isnan(extra[blocks==0]).all()
    for split,x,frame in [('fit',extra,hist),('valid',vx,evalrows)]:np.save(OUT/f'{split}.npy',x);np.save(OUT/f'y_{split}.npy',frame['watched'].to_numpy())
    np.save(OUT/'ids.npy',ids);print('prepared',extra.shape,vx.shape,flush=True)


class Batches(xgb.DataIter):
    def __init__(self,split):
        self.x=np.load(M/f'{split}.npy',mmap_mode='r'); self.e=np.load(OUT/f'{split}.npy',mmap_mode='r');self.y=np.load(OUT/f'y_{split}.npy');assert len(self.x)==len(self.e)==len(self.y);self.reset();super().__init__()
    def reset(self):self.i=0
    def next(self,input_data):
        if self.i>=len(self.y):return False
        sl=slice(self.i,self.i+8192);self.i+=8192;input_data(data=np.column_stack([self.x[sl],self.e[sl]]),label=self.y[sl]);return True


def main():
    OUT.mkdir(exist_ok=True,parents=True);start=time.monotonic();prepare();ids=np.load(OUT/'ids.npy');y=np.load(OUT/'y_valid.npy');z=np.load(ROOT/'artifacts/local-t016/validation_predictions.npz');order=z['row_ids'].argsort();pos=order[np.searchsorted(z['row_ids'][order],ids)];np.testing.assert_array_equal(z['row_ids'][pos],ids);np.testing.assert_array_equal(z['y_true'].ravel()[pos],y);p0=z['probabilities'].ravel()[pos];b=np.load(M/'predictions.npz');np.testing.assert_array_equal(ids,b['row_ids']); ref=b['blend'];np.testing.assert_allclose(ref,.75*p0+.25*b['probabilities'],atol=1e-7)
    fit=xgb.QuantileDMatrix(Batches('fit'),max_bin=127,nthread=6);va=xgb.QuantileDMatrix(Batches('valid'),ref=fit,max_bin=127,nthread=6);model=xgb.train({**BASE,**CONFIGS['depth12_node'],'device':'cpu','nthread':6},fit,900,evals=[(va,'holdout')],verbose_eval=100);pred=model.predict(va);blend=.75*p0+.25*pred
    sample=np.column_stack([np.load(M/'valid.npy',mmap_mode='r')[:16],np.load(OUT/'valid.npy',mmap_mode='r')[:16]]);np.testing.assert_array_equal(model.inplace_predict(sample),np.concatenate([model.inplace_predict(row[None,:]) for row in sample]));np.testing.assert_array_equal(model.inplace_predict(sample),model.inplace_predict(sample[::-1].copy())[::-1]);np.testing.assert_allclose(model.inplace_predict(sample),pred[:16],atol=1e-6)
    auc=roc_auc_score(y,blend);report=dict(standalone_auc=roc_auc_score(y,pred),fixed_blend_auc=auc,delta_vs_T059=auc-roc_auc_score(y,ref),delta_vs_T016=auc-roc_auc_score(y,p0),seconds=time.monotonic()-start);model.save_model(OUT/'model.ubj');np.savez_compressed(OUT/'predictions.npz',row_ids=ids,probabilities=pred,blend=blend);(TRIAL/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)


if __name__=='__main__':main()
