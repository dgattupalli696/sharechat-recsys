"""Run from the repository root. Each CPU fit runs in its own process."""
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
import polars as pl
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from embed_screen import unit_vectors
from feature_screen import CUTOFF
from xgb_tune import BASE, CONFIGS

M = ROOT / 'artifacts/local-t036/matrices'
OUT = ROOT / 'artifacts/local-t054'
TRIAL = Path(__file__).resolve().parent
THREADS = 6


def load(name):
    return np.load(M / f'{name}.npy', mmap_mode='r')


def prepare():
    f = pl.read_parquet('data/train.parquet', columns=['request_id', 'feature1', 'feature18', 'feature23', 'feature32', 'feature43', 'feature54', 'watched'])
    valid = (f['feature1'] >= CUTOFF).to_numpy()
    assert np.array_equal(np.flatnonzero(valid), load('row_ids'))
    v, u = (unit_vectors(f[c]) for c in ('feature43', 'feature32'))
    a = (v*u).sum(1)
    col = json.loads((M/'columns.json').read_text())['columns'].index('cos_32_43')
    for mask, part in [(~valid, 'fit'), (valid, 'valid')]:
        np.testing.assert_array_equal(f['watched'].to_numpy()[mask], load('y_'+part))
        np.testing.assert_allclose(a[mask], load('x_'+part)[:,col], atol=2e-6)
    usable = np.any(v != 0, axis=1)
    a[~(usable & np.any(u != 0, axis=1))] = np.nan
    del u
    d = f['feature54'].to_numpy()
    # Sort once; process equal-size requests together with bounded temporary matrices.
    s = f.select('request_id','feature18','feature23').with_row_index('idx').sort(['request_id','feature18','idx'])
    item, creator = (s[c].cast(pl.Categorical).to_physical().to_numpy() for c in ['feature18','feature23'])
    ix = s['idx'].to_numpy(); sizes = s.group_by('request_id', maintain_order=True).len()['len'].to_numpy()
    starts = np.cumsum(sizes, dtype=np.int64)-sizes
    extra = np.full((len(f), 4), np.nan, np.float32)
    for n in np.unique(sizes):
        begins = starts[sizes == n]
        for off in range(0,len(begins),2048):
            pos = begins[off:off+2048,None]+np.arange(n); rows = ix[pos]
            it, cr = item[pos], creator[pos]; other = it[:,:,None] != it[:,None,:]
            same = other & (cr[:,:,None] == cr[:,None,:]); first = np.ones_like(it, bool)
            first[:,1:] = it[:,1:] != it[:,:-1]
            count = (same & first[:,None,:]).sum(2)
            best = np.where(same & np.isfinite(a[rows])[:,None,:], a[rows][:,None,:], -np.inf).max(2)
            gap = a[rows] - best; gap[~np.isfinite(gap)] = np.nan
            emb = v[rows]; sim = emb @ emb.transpose(0,2,1)
            ok = other & usable[rows][:,:,None] & usable[rows][:,None,:]; sim[~ok] = -np.inf
            nearest = np.take_along_axis(rows, sim.argmax(2), axis=1)
            ag = a[rows]-a[nearest]; dg = d[rows]-d[nearest]; ag[~ok.any(2)] = dg[~ok.any(2)] = np.nan
            extra[rows] = np.stack([count,gap,ag,dg], axis=2)
            if off == 0:  # Independent scalar reference, including singleton and ties.
                for j, r in enumerate(rows[0]):
                    rivals = [int(x) for k,x in enumerate(rows[0]) if it[0,k] != it[0,j]]
                    peers = [x for x in rivals if f['feature23'][x] == f['feature23'][int(r)]]
                    opts = [x for x in rivals if usable[r] and usable[x]]
                    # Near-ties can round differently between GEMM and scalar dots.
                    if opts and np.sum(np.isclose([v[r]@v[x] for x in opts],max(v[r]@v[x] for x in opts), atol=1e-6)) > 1:
                        continue
                    q = max(opts,key=lambda x:v[r]@v[x]) if opts else None
                    expect = [len({f['feature18'][x] for x in peers}), a[r]-max((a[x] for x in peers if np.isfinite(a[x])),default=np.nan), a[r]-a[q] if q is not None else np.nan, d[r]-d[q] if q is not None else np.nan]
                    np.testing.assert_allclose(extra[r],expect,atol=2e-6,equal_nan=True)
    for mask,part in [(~valid,'fit'),(valid,'valid')]:
        np.save(OUT/f'extra_{part}.npy', extra[mask]); print(part,'coverage',np.isfinite(extra[mask]).mean(0),flush=True)
    np.save(OUT/'groups.npy',f['request_id'].filter(pl.Series(valid)).cast(pl.Categorical).to_physical().to_numpy())


class Batches(xgb.DataIter):
    def __init__(self, split, width, prefix='extra'):
        self.x, self.y = load('x_'+split), load('y_'+split)
        self.e = np.load(OUT/f'{prefix}_{split}.npy',mmap_mode='r')[:,:width]; self.width=width
        self.reset(); super().__init__()

    def reset(self):
        self.i=0

    def next(self, input_data):
        if self.i >= len(self.y): return False
        sl=slice(self.i,self.i+8192); self.i+=8192
        x=np.column_stack([self.x[sl],self.e[sl]]) if self.width else self.x[sl]
        input_data(data=x,label=self.y[sl]); return True


def peerscore():
    ids, y = load('row_ids'), load('y_valid'); total=np.zeros(len(y))
    paths=['024/t024/ensemble-32/validation_predictions','027/t027/ensemble-32/validation_predictions','036/t036/leaves511_eta02-holdout','041/t041/catboost_ids-holdout','040/t040/cross-masked-fusion-cosine/validation_predictions']
    for path in paths:
        with np.load(f'artifacts/kaggle-t{path}.npz') as z:
            order=np.argsort(z['row_ids']); pos=order[np.searchsorted(z['row_ids'][order],ids)]
            np.testing.assert_array_equal(z['row_ids'][pos],ids); np.testing.assert_array_equal(z['y_true'].ravel()[pos],y)
            total+=rankdata(z['probabilities'].ravel()[pos])
    return total


def fit(arm):
    began=time.monotonic(); width={'A':0,'B':1,'C':4,'D':2}[arm]; peers=peerscore()
    params={**BASE,**CONFIGS['depth12_node'],'device':'cpu','nthread':THREADS}
    prefix='match' if arm=='D' else 'extra'
    fit=xgb.QuantileDMatrix(Batches('fit',width,prefix),max_bin=127,nthread=THREADS)
    valid=xgb.QuantileDMatrix(Batches('valid',width,prefix),max_bin=127,nthread=THREADS,ref=fit)
    model=xgb.train(params,fit,20000,evals=[(valid,'holdout')],early_stopping_rounds=300,verbose_eval=50)
    pred=model.predict(valid,iteration_range=(0,model.best_iteration+1)); blend=peers+rankdata(pred); y=load('y_valid')
    result=dict(arm=arm, standalone_auc=roc_auc_score(y,pred),blend_auc=roc_auc_score(y,blend),best_round=model.best_iteration+1,seconds=time.monotonic()-began,peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**30)
    np.savez_compressed(OUT/f'{arm}.npz',row_ids=load('row_ids'),probabilities=pred,blend=blend)
    (TRIAL/f'{arm}.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)


def compare(match=False):
    groups=np.unique(np.load(OUT/'groups.npy'),return_inverse=True)[1]; y=load('y_valid'); rng=np.random.default_rng(42)
    # Weighted, tie-aware pairwise AUC via positive/negative counts per score.
    orders=[]
    for arm in ('AD' if match else 'ABC'):
        score=np.load(OUT/f'{arm}.npz')['blend']; ix=score.argsort(); starts=np.r_[0,np.flatnonzero(np.diff(score[ix]))+1]; orders.append((ix,starts))
    diffs=[]; n=groups.max()+1
    for _ in range(400):
        w=np.bincount(rng.integers(n,size=n),minlength=n)[groups]; auc=[]
        for ix,starts in orders:
            pos=np.add.reduceat((w*y)[ix],starts); neg=np.add.reduceat((w*(1-y))[ix],starts); auc.append(pos@(neg.cumsum()-.5*neg)/(float(pos.sum())*neg.sum()))
        diffs.append([auc[1]-auc[0]] if match else [auc[2]-auc[0],auc[2]-auc[1],auc[1]-auc[0]])
    summary=dict(zip(['D-A'] if match else ['C-A','C-B','B-A'],np.quantile(diffs,[.025,.975],axis=0).T.tolist()));(TRIAL/('match-bootstrap.json' if match else 'bootstrap.json')).write_text(json.dumps(summary,indent=2)+'\n');print(summary)


if __name__=='__main__':
    os.chdir(ROOT); OUT.mkdir(exist_ok=True,parents=True)
    if len(sys.argv)>1:
        if sys.argv[1]=='prepare': prepare()
        else: fit(sys.argv[1])
    else:
        for stage in ['prepare',*'ABC']:
            done=OUT/'groups.npy' if stage=='prepare' else TRIAL/f'{stage}.json'
            if not done.exists(): subprocess.run([sys.executable,'-u',__file__,stage],check=True)
        compare()
