"""Confirm primary winners on T018's earlier fold and matched neural anchor."""
import importlib.util
import json
from datetime import datetime
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

spec=importlib.util.spec_from_file_location('priors',TRIAL.parent/'060-row-only-audience-priors/run.py'); priors=importlib.util.module_from_spec(spec);spec.loader.exec_module(priors)
M=ROOT/'artifacts/local-t059';OUT=ROOT/'artifacts/local-t061';BUDGET=900


def anchor(fold,ids,y):
    with np.load(ROOT/f'artifacts/local-t018-confirmation-analysis/{fold}/frozen-75-25.npz') as z:
        order=z['row_ids'].argsort();pos=order[np.searchsorted(z['row_ids'][order],ids)];np.testing.assert_array_equal(z['row_ids'][pos],ids);np.testing.assert_array_equal(z['y_true'][pos].ravel(),y);return z['probabilities'][pos].ravel()


def masks(f):
    t=pl.col('feature1');return [(f.select(cond).to_series().to_numpy()) for cond in [t<datetime(2026,10,27,8,6,52,397000),(t>=datetime(2026,10,27,8,6,52,397000))&(t<datetime(2026,10,27,18)),t<CUTOFF]]


def ci(y,a,b,groups):
    # Paired request-cluster bootstrap. IDs are evaluation metadata only.
    info=[]
    for s in [a,b]:
        ix=s.argsort();info.append((ix,np.r_[0,np.flatnonzero(np.diff(s[ix]))+1]))
    rng=np.random.default_rng(42);n=groups.max()+1;d=[]
    for _ in range(400):
        w=np.bincount(rng.integers(n,size=n),minlength=n)[groups];auc=[]
        for ix,k in info:
            p=np.add.reduceat((w*y)[ix],k);q=np.add.reduceat((w*(1-y))[ix],k);auc.append(p@(q.cumsum()-.5*q)/(float(p.sum())*q.sum()))
        d.append(auc[1]-auc[0])
    return np.quantile(d,[.025,.975]).tolist()


def main():
    OUT.mkdir(exist_ok=True,parents=True);start=time.monotonic();f=pl.read_parquet(ROOT/'data/train.parquet',columns=['feature1','watched',*priors.COLS]);tr,va,pre=masks(f);assert (tr.sum(),va.sum(),pre.sum())==(710330,299969,1230110)
    indices=[np.flatnonzero(m[pre]) for m in (tr,va)]; ids=np.flatnonzero(va);y=f['watched'].to_numpy();target=y[va];ref=anchor('earlier',ids,target);auc0=roc_auc_score(target,ref);assert abs(auc0-.8375490212)<1e-9; primary=anchor('primary',np.flatnonzero(~pre),y[~pre]);pauc=roc_auc_score(y[~pre],primary);results={};x=np.load(M/'fit.npy',mmap_mode='r')
    groups=np.unique(pl.read_parquet(ROOT/'data/train.parquet',columns=['request_id'])['request_id'].filter(pl.Series(va)).to_numpy(),return_inverse=True)[1]
    for arm in ['059','060']:
        with np.load(ROOT/f'artifacts/local-t{arm}/predictions.npz') as z:
            np.testing.assert_array_equal(np.flatnonzero(~pre),z['row_ids']);p=z['probabilities'];original=roc_auc_score(y[~pre],z['blend'])-.8408057533
        matched=roc_auc_score(y[~pre],.75*primary+.25*p)-pauc; record={'primary_gain_vs_T016':original,'primary_gain_vs_matched_T018':matched};results[arm]=record
        if original<.0003 or (arm=='060' and original<=results['059']['primary_gain_vs_T016']):record['decision']='did not clear primary gate';continue
        extra=[None,None]
        if arm=='060':
            h=f.filter(pl.Series(tr));tf=h['feature1'].cast(pl.Int64).to_numpy();blocks=np.searchsorted(np.quantile(tf,[.1,.2,.4,.6,.8],method='inverted_cdf'),tf,side='right');fx=np.full((len(h),2*len(priors.KEYS)),np.nan,'f4');
            for k in range(1,6):fx[blocks==k]=priors.apply(h.filter(pl.Series(blocks==k)).select(priors.COLS),priors.learn(h.filter(pl.Series(blocks<k))))
            extra=[fx,priors.apply(f.filter(pl.Series(va)).select(priors.COLS),priors.learn(h))]
        class Batch(xgb.DataIter):
            def __init__(self,i):self.i=i;self.reset();super().__init__()
            def reset(self):self.at=0
            def next(self,input_data):
                if self.at>=len(indices[self.i]):return False
                s=slice(self.at,self.at+8192);self.at+=8192;idx=indices[self.i][s];features=x[idx];e=extra[self.i];input_data(data=features if e is None else np.column_stack([features,e[s]]),label=labels[self.i][s]);return True
        labels=[y[tr],target];fit=xgb.QuantileDMatrix(Batch(0),max_bin=127,nthread=6);val=xgb.QuantileDMatrix(Batch(1),ref=fit,max_bin=127,nthread=6);model=xgb.train({**BASE,**CONFIGS['depth12_node'],'device':'cpu','nthread':6},fit,BUDGET,evals=[(val,'earlier')],verbose_eval=200);pred=model.predict(val);blend=.75*ref+.25*pred;auc=roc_auc_score(target,blend);bounds=ci(target,ref,blend,groups);record.update(earlier_anchor=auc0,earlier_standalone=roc_auc_score(target,pred),earlier_blend=auc,earlier_gain=auc-auc0,paired_95pct=bounds,decision='confirmed' if matched>0 and bounds[0]>0 else 'not confirmed');np.savez_compressed(OUT/f'{arm}.npz',row_ids=ids,predictions=pred,blend=blend);print(arm,record,flush=True);del fit,val,model
    results['seconds']=time.monotonic()-start;(TRIAL/'results.json').write_text(json.dumps(results,indent=2)+'\n');print(results,flush=True)


if __name__=='__main__':main()
