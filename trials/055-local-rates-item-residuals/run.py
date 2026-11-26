"""T055: two independent feature ablations against the archived local T054-A."""
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
import xgboost as xgb

TRIAL = Path(__file__).resolve().parent
ROOT = TRIAL.parents[1]
OUT = ROOT / 'artifacts/local-t055'
spec = importlib.util.spec_from_file_location('t054', TRIAL.parent/'054-local-xgboost-rivals/run.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def write(name, obj):
    (TRIAL/f'{name}.json').write_text(json.dumps(obj, indent=2)+'\n')
    print(name, obj, flush=True)


def interval_features(den, nums):
    dn = np.diff(den, axis=1)
    features = []
    for num in nums:
        delta = np.diff(num, axis=1)
        sane = (den >= 0) & (num >= 0) & (num <= den) & np.isfinite(den) & np.isfinite(num)
        ok = sane[:, :-1] & sane[:, 1:] & (dn > 0) & (delta >= 0) & (delta <= dn)
        features.append(np.divide(delta, dn, where=ok, out=np.full_like(delta, np.nan)))
    return np.column_stack(features).astype(np.float32)


def prepare_rates():
    names = json.loads((base.M/'columns.json').read_text())['columns']
    coverage = {}
    for split in ('fit','valid'):
        x = base.load('x_'+split)
        def block(start):
            return x[:, [names.index(f'feature{i}') for i in range(start,start+4)]].astype('d')
        den = block(189); nums = [block(213), block(221)]
        for num, start in zip(nums,[408,416]):
            rate = block(start); ok = (den>0) & np.isfinite(num) & np.isfinite(rate)
            # Allow for quantization of larger raw counts in the cached float32 matrix.
            np.testing.assert_allclose((num[ok]/den[ok]), rate[ok], rtol=2e-6, atol=1e-8)
        e = interval_features(den, nums); np.save(OUT/f'rates_{split}.npy', e)
        coverage[split] = np.isfinite(e).mean(0).tolist()
    write('rate-coverage', coverage)


def prior(item, t, residual):
    """Strictly earlier same-item sums/counts; NaNs never contribute (including holdout)."""
    order = np.lexsort((t,item)); i,ts,r = item[order],t[order],residual[order]
    group = np.r_[True,(i[1:]!=i[:-1]) | (ts[1:]!=ts[:-1])]; start = np.flatnonzero(group)
    itemstart = np.flatnonzero(np.r_[True,i[1:]!=i[:-1]])
    at = np.cumsum(group)-1; origin = itemstart[np.cumsum(np.r_[True,i[1:]!=i[:-1]])-1]
    sums = np.r_[0,np.cumsum(np.nan_to_num(r,nan=0.),dtype='d')]; counts=np.r_[0,np.isfinite(r).cumsum()]
    n = counts[start[at]]-counts[origin]; v=(sums[start[at]]-sums[origin])/(n+20); v[n==0]=np.nan
    result=np.empty(len(order),dtype='f4'); result[order]=v; return result


def prepare_history(return_residual=False):
    f=pl.read_parquet(ROOT/'data/train.parquet',columns=['feature1','feature18','watched'])
    valid=(f['feature1']>=base.CUTOFF).to_numpy(); np.testing.assert_array_equal(np.flatnonzero(valid),base.load('row_ids'))
    item=f['feature18'].cast(pl.Categorical).to_physical().to_numpy(); t=f['feature1'].cast(pl.Int64).to_numpy(); y=base.load('y_fit')
    np.testing.assert_array_equal(y,f['watched'].to_numpy()[~valid]); tf=t[~valid]
    names=json.loads((base.M/'columns.json').read_text())['columns']
    teacher=['rq_size','cos_32_43','cos_30_46']+[f'feature{i}' for i in [3,54,60,189,190,191,192,408,409,410,411,416,417,418,419,430,431,475,476,477,478,479,480,483,484,492,493,506]]+[f'rq_{mode}_feature{i}' for mode in ('rank','dev','gap') for i in (418,419,411,410,430,431,417,352,346,480)]
    x=np.ascontiguousarray(base.load('x_fit')[:,[names.index(c) for c in teacher]])
    edges=np.quantile(tf,[.2,.4,.6,.8],method='inverted_cdf'); block=np.searchsorted(edges,tf,side='right'); pred=np.full(len(y),np.nan); audits=[]
    params=dict(objective='binary',learning_rate=.05,num_leaves=63,min_data_in_leaf=150,max_bin=127,num_threads=6,seed=42,verbosity=-1,force_col_wise=True,deterministic=True)
    for k in range(1,5):
        fit=block<k; score=block==k; assert tf[fit].max()<tf[score].min()
        model=lgb.train(params,lgb.Dataset(x[fit],label=y[fit]),num_boost_round=180)
        pred[score]=model.predict(x[score],num_threads=6)
        audits.append(dict(block=k,fit=int(fit.sum()),scored=int(score.sum()),forward_auc=float(roc_auc_score(y[score],pred[score]))));print(audits[-1],flush=True)
    r=np.full(len(t),np.nan); r[~valid]=y-pred
    if return_residual: return r
    h=prior(item,t,r)
    # Check scalar histories on real rows without using any held-out labels.
    for j in np.random.default_rng(42).choice(len(t),32,replace=False):
        values=r[(item==item[j]) & (t<t[j]) & np.isfinite(r)]
        np.testing.assert_allclose(h[j],values.sum()/(20+len(values)) if len(values) else np.nan,atol=1e-7,equal_nan=True)
    for mask,split in [(~valid,'fit'),(valid,'valid')]: np.save(OUT/f'history_{split}.npy',h[mask,None])
    write('history-audit',dict(blocks=audits,teacher_columns=teacher,coverage_fit=float(np.isfinite(h[~valid]).mean()),coverage_valid=float(np.isfinite(h[valid]).mean())))


class Batches(base.Batches):
    def __init__(self, split, arm):
        self.x,self.y=base.load('x_'+split),base.load('y_'+split); self.e=np.load(OUT/f'{arm}_{split}.npy',mmap_mode='r'); self.width=self.e.shape[1]
        assert len(self.e)==len(self.y); self.reset(); xgb.DataIter.__init__(self)


def interval(y,scores):
    groups=np.unique(np.load(base.OUT/'groups.npy'),return_inverse=True)[1]; n=groups.max()+1; rng=np.random.default_rng(42)
    orders=[]
    for s in scores:
        ix=np.argsort(s); starts=np.r_[0,np.flatnonzero(np.diff(s[ix]))+1]; orders.append((ix,starts))
    diffs=[]
    for _ in range(400):
        w=np.bincount(rng.integers(n,size=n),minlength=n)[groups]; auc=[]
        for ix,starts in orders:
            p=np.add.reduceat((w*y)[ix],starts); q=np.add.reduceat((w*(1-y))[ix],starts); auc.append(p@(q.cumsum()-.5*q)/(float(p.sum())*q.sum()))
        diffs.append(auc[1]-auc[0])
    return np.quantile(diffs,[.025,.975]).tolist()


def train(arm):
    began=time.monotonic(); y=base.load('y_valid'); ids=base.load('row_ids'); peers=base.peerscore()
    with np.load(base.OUT/'A.npz') as z:
        np.testing.assert_array_equal(ids,z['row_ids']); reference=z['blend']; np.testing.assert_array_equal(reference,peers+rankdata(z['probabilities']))
    ref_auc=roc_auc_score(y,reference); assert abs(ref_auc-.8910758705604122)<1e-10
    params={**base.BASE,**base.CONFIGS['depth12_node'],'device':'cpu','nthread':6}
    fit=xgb.QuantileDMatrix(Batches('fit',arm),max_bin=127,nthread=6)
    valid=xgb.QuantileDMatrix(Batches('valid',arm),max_bin=127,nthread=6,ref=fit)
    model=xgb.train(params,fit,20000,evals=[(valid,'holdout')],early_stopping_rounds=300,verbose_eval=50)
    pred=model.predict(valid,iteration_range=(0,model.best_iteration+1)); blend=peers+rankdata(pred); auc=roc_auc_score(y,blend)
    np.savez_compressed(OUT/f'{arm}.npz',row_ids=ids,probabilities=pred,blend=blend)
    write(arm,dict(standalone_auc=roc_auc_score(y,pred),blend_auc=auc,delta_vs_A=auc-ref_auc,request_bootstrap_95pct=interval(y,[reference,blend]),best_round=model.best_iteration+1,seconds=time.monotonic()-began,peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30))


if __name__=='__main__':
    os.chdir(ROOT); OUT.mkdir(exist_ok=True,parents=True)
    if len(sys.argv)>1:
        s=sys.argv[1]
        if s=='prepare-rates': prepare_rates()
        elif s=='prepare-history': prepare_history()
        elif s in ('rates','history'): train(s)
        else: raise ValueError(s)
    else:
        for stage,done in [('prepare-rates','rate-coverage'),('prepare-history','history-audit'),('rates','rates'),('history','history')]:
            if not (TRIAL/f'{done}.json').exists(): subprocess.run([sys.executable,'-u',__file__,stage],check=True)
