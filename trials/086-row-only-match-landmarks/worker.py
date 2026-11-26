"""Two-fold row-only tree diversity on a compact view of user/video embedding matches."""
from pathlib import Path
import sys,time,json
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import base62 as base
from rich63 import rich
from t069 import duration
import landmarks86

OUT=Path('/kaggle/working/t086');CACHE=Path('/kaggle/temp/t086');BASE=594+36+49;WIDTH=BASE+320

def products(f):
    def unit(k):
        x=f[f'feature{k}'].fill_null([0.]*32).list.to_array(32).to_numpy().astype('f4');return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-6)
    v={k:unit(k) for pair in base.PAIRS for k in pair};return np.column_stack([v[u]*v[i] for u,i in base.PAIRS])

def prepare(raw):
    names=base.numeric(pl.scan_parquet(raw).collect_schema());fields=names+[f'feature{i}' for i in [*range(26,33),*range(40,47)]];assert 'request_id' not in fields
    pf=pq.ParquetFile(raw);x=np.lib.format.open_memmap(CACHE/'x.npy',mode='w+',dtype='f4',shape=(1537638,BASE));off=0
    vec=np.lib.format.open_memmap(CACHE/'vec.npy',mode='w+',dtype='f4',shape=(1537638,14,32))
    for batch in pf.iter_batches(batch_size=8192,columns=fields):
        f=pl.from_arrow(batch);vec[off:off+len(f)]=np.stack([f[f'feature{k}'].fill_null([0.]*32).list.to_array(32).to_numpy() for k in [*range(26,33),*range(40,47)]],axis=1);x[off:off+len(f)]=np.column_stack([base.transform(f,names),duration(f),rich(f)[:,448:497]]);off+=len(f)
    assert off==len(x);x.flush();f=pl.read_parquet(raw,columns=['feature1','watched']);np.save(CACHE/'t.npy',f['feature1'].to_numpy());np.save(CACHE/'y.npy',f['watched'].to_numpy())

def fit(fold,arm):
    from sklearn.metrics import roc_auc_score
    t=np.load(CACHE/'t.npy');y=np.load(CACHE/'y.npy');x=np.load(CACHE/'x.npy',mmap_mode='r');cut=base.CUTOFF if fold=='primary' else np.datetime64('2026-10-27T08:06:52.397');tr=np.flatnonzero(t<cut);va=np.flatnonzero((t>=cut)&((fold=='primary')|(t<np.datetime64('2026-10-27T18:00'))));assert (len(tr),len(va))==((1230110,307528) if fold=='primary' else (710330,299969));width=BASE;bins=127;leaf=False;vec=np.load(CACHE/'vec.npy',mmap_mode='r');state=landmarks86.learn(vec,tr,arm);extra=np.empty((len(vec),72),'f4')
    for start in range(0,len(vec),8192):extra[start:start+8192]=landmarks86.apply(vec[start:start+8192],state)
    def matrix(j):return np.column_stack([x[j,:width],extra[j]])
    if arm.startswith('cat'):
        from catboost import CatBoostClassifier, Pool
        pool=Pool(np.ascontiguousarray(x[tr,:width]),label=y[tr],thread_count=4);model=CatBoostClassifier(iterations=1600,learning_rate=.035,depth=7,l2_leaf_reg=8,border_count=128,random_seed=42,task_type='GPU',devices='0',gpu_ram_part=.85,verbose=400,allow_writing_files=False);model.fit(pool);del pool
        def predict(j):return model.predict_proba(np.ascontiguousarray(x[j,:width]),thread_count=4)[:,1]
    else:
        import cupy as cp
        import xgboost as xgb
        class Iter(xgb.DataIter):
            def __init__(self):self.reset();super().__init__()
            def reset(self):self.i=0
            def next(self,input_data):
                if self.i>=len(tr):return False
                j=tr[self.i:self.i+8192];self.i+=8192;input_data(data=cp.asarray(matrix(j)),label=cp.asarray(y[j]));return True
        dm=xgb.QuantileDMatrix(Iter(),nthread=4,max_bin=bins);params=dict(objective='binary:logistic',tree_method='hist',device='cuda',max_depth=0 if leaf else 8,max_leaves=128 if leaf else 0,grow_policy='lossguide' if leaf else 'depthwise',eta=.025,subsample=.8,reg_lambda=10,seed=42,max_bin=bins,colsample_bytree=.65,colsample_bynode=.5,min_child_weight=100,gamma=1,nthread=4);model=xgb.train(params,dm,1600);del dm
        def predict(j):return cp.asnumpy(model.inplace_predict(cp.asarray(matrix(j))))
    p=np.concatenate([predict(va[i:i+8192]) for i in range(0,len(va),8192)]);np.testing.assert_allclose(p[:8],np.concatenate([predict(j[None]) for j in va[:8]]),atol=1e-7);assert np.isfinite(p).all();np.savez_compressed(OUT/f'{fold}-{arm}.npz',row_ids=va,y_true=y[va],probabilities=p);return {'auc':roc_auc_score(y[va],p),'features':width}

if __name__=='__main__':
    OUT.mkdir(exist_ok=True,parents=True);CACHE.mkdir(exist_ok=True,parents=True);args=sys.argv[1:];start=time.monotonic()
    if args==['prepare']:prepare(next(p for p in Path('/kaggle/input').rglob('train.parquet') if (p.parent/'sample_submission.csv').exists()));res={}
    else:res=fit(*args)
    res['seconds']=time.monotonic()-start;(OUT/('-'.join(args)+'.json')).write_text(json.dumps(res)+'\n');print(args,res,flush=True)
