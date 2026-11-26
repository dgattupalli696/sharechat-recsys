"""Offline matched feature-transfer test. No test inference or submission."""
from pathlib import Path
import importlib.util,json,sys,time,os
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import xgboost as xgb
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

T=Path(__file__).resolve().parent;ROOT=T.parents[1];OUT=ROOT/'artifacts/local-t089';os.chdir(ROOT)
def module(name,rel):
    s=importlib.util.spec_from_file_location(name,T.parent/rel);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
old=module('t054','054-local-xgboost-rivals/run.py');module('base62','062-row-only-gpu-replays/worker.py');module('rich63','063-row-only-embedding-depth/worker.py');dur=module('t069','069-row-only-duration-budget/worker.py');rot=module('rot83','083-row-only-paired-rotations/rotations.py');ci=module('t061','061-row-only-temporal-confirmation/run.py').ci

def prepare():
    raw=ROOT/'data/train.parquet';meta=pl.read_parquet(raw,columns=['feature1','watched']);v=(meta['feature1']>=old.CUTOFF).to_numpy();np.testing.assert_array_equal(np.flatnonzero(v),old.load('row_ids'));labels=meta['watched'].to_numpy();selected=np.sort(np.random.default_rng(42).choice(np.flatnonzero(~v),32768,replace=False));cols=[f'feature{i}' for i in [*range(26,33),*range(40,47)]];fields=cols+[f'feature{i}' for i in [54,*dur.H]];pf=pq.ParquetFile(raw)
    def vec(f):return np.stack([f[k].fill_null([0.]*32).list.to_array(32).to_numpy() for k in cols],axis=1)
    # Learn on the same training-prefix sample as T083 without caching a 2.7 GB raw vector array.
    samples=[];off=0
    for batch in pf.iter_batches(columns=cols,batch_size=8192):
        end=off+len(batch);lo,hi=np.searchsorted(selected,[off,end]);
        if hi>lo:samples.append(vec(pl.from_arrow(batch.take(selected[lo:hi]-off))))
        off=end
    sample=np.concatenate(samples);assert len(sample)==32768;state=rot.learn(sample,np.arange(len(sample)),False);del sample,samples
    targets={};pos={}
    for part,mask in [('fit',~v),('valid',v)]:
        np.testing.assert_array_equal(labels[mask],old.load('y_'+part));targets[part]=np.lib.format.open_memmap(OUT/f'{part}.npy',mode='w+',dtype='f4',shape=(int(mask.sum()),96));pos[part]=0
    off=0
    for batch in pf.iter_batches(columns=fields,batch_size=8192):
        f=pl.from_arrow(batch);z=np.column_stack([dur.duration(f),rot.apply(vec(f),state)]);vs=v[off:off+len(f)]
        for part,mask in [('fit',~vs),('valid',vs)]:
            start=pos[part];pos[part]+=mask.sum();targets[part][start:pos[part]]=z[mask]
        off+=len(f)
    assert off==len(meta) and pos=={'fit':1230110,'valid':307528};print('validated row alignment and built 36 duration + 60 rotation features',flush=True)

class Batches(xgb.DataIter):
    def __init__(self,part,sl):
        self.x=old.load('x_'+part);self.e=np.load(OUT/f'{part}.npy',mmap_mode='r')[:,sl];self.y=old.load('y_'+part);self.reset();super().__init__()
    def reset(self):self.i=0
    def next(self,input_data):
        if self.i>=len(self.y):return False
        s=slice(self.i,self.i+8192);self.i+=8192;input_data(data=np.column_stack([self.x[s],self.e[s]]),label=self.y[s]);return True

def fit(name,sl):
    y=old.load('y_valid');ref=np.load(old.OUT/'A.npz');np.testing.assert_array_equal(ref['row_ids'],old.load('row_ids'));peers=old.peerscore();np.testing.assert_array_equal(ref['blend'],peers+rankdata(ref['probabilities']));a=roc_auc_score(y,ref['blend']);assert abs(a-.89107587056)<1e-9
    train=xgb.QuantileDMatrix(Batches('fit',sl),max_bin=127,nthread=6);valid=xgb.QuantileDMatrix(Batches('valid',sl),ref=train,max_bin=127,nthread=6);model=xgb.train({**old.BASE,**old.CONFIGS['depth12_node'],'device':'cpu','nthread':6},train,20000,evals=[(valid,'holdout')],early_stopping_rounds=300,verbose_eval=200);p=model.predict(valid,iteration_range=(0,model.best_iteration+1));b=peers+rankdata(p);groups=np.unique(np.load(old.OUT/'groups.npy'),return_inverse=True)[1];score=roc_auc_score(y,b);r=dict(standalone_auc=roc_auc_score(y,p),blend_auc=score,delta=score-a,paired_95pct=ci(y,ref['blend'],b,groups),best_round=model.best_iteration+1);(T/f'{name}.json').write_text(json.dumps(r,indent=2)+'\n');print(name,r,flush=True)

if __name__=='__main__':
    import subprocess
    OUT.mkdir(exist_ok=True,parents=True);arms={'duration':slice(0,36),'rotation':slice(36,96),'both':slice(None)}
    if len(sys.argv)>1:fit(sys.argv[1],arms[sys.argv[1]])
    else:
        prepare()
        for arm in arms:subprocess.run([sys.executable,'-u',__file__,arm],check=True)
