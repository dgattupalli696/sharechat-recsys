"""T063: paired depth / richer same-row embedding experiment, two chronological folds."""
import json
from pathlib import Path
import sys
import time
import numpy as np
import polars as pl
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score
import base62 as base

OUT=Path('/kaggle/working/t066');CACHE=Path('/kaggle/temp/t066');EXTRA=448+49+14


def rich(f):
    vec=np.stack([f[f'feature{i}'].fill_null([0.]*32).list.to_array(32).to_numpy() for i in [*range(26,33),*range(40,47)]],axis=1).astype('f4');bias=np.column_stack([f[f'feature{i}'].to_numpy() for i in [*range(33,40),*range(47,54)]])
    # Bias interpretation is a hypothesis, not a supplied field definition.
    interactions=np.einsum('bik,bjk->bij',vec[:,:7],vec[:,7:])+bias[:,:7,None]+bias[:,None,7:];return np.column_stack([vec.reshape(len(f),448),interactions.reshape(len(f),49),np.linalg.norm(vec,axis=2)])


def prepare(data, split):
    raw=data/f'{split}.parquet';names=base.numeric(pl.scan_parquet(raw).collect_schema());fields=names+[f'feature{i}' for i in [*range(26,33),*range(40,47)]];n=pq.ParquetFile(raw).metadata.num_rows;assert n==(1537638 if split=='train' else 1037090) and len(names)==584 and len(set(fields))==len(fields)
    m=np.lib.format.open_memmap(CACHE/f'{split}.npy',mode='w+',dtype='f4',shape=(n,594+EXTRA));off=0
    for batch in pq.ParquetFile(raw).iter_batches(columns=fields,batch_size=8192):
        f=pl.from_arrow(batch);x=np.column_stack([base.transform(f,names),rich(f)]);m[off:off+len(f)]=x;off+=len(f)
    assert off==n;m.flush()
    if split=='test':return
    f=pl.read_parquet(raw,columns=['feature1','watched']);np.save(CACHE/'t.npy',f['feature1'].to_numpy());np.save(CACHE/'y.npy',f['watched'].to_numpy())


def fit(fold,depth,more):
    import cupy as cp
    import xgboost as xgb
    t=np.load(CACHE/'t.npy');y=np.load(CACHE/'y.npy');x=np.load(CACHE/'train.npy',mmap_mode='r');tr=np.arange(len(y));assert len(tr)==1537638; width=x.shape[1] if more else 594
    class Batch(xgb.DataIter):
        def __init__(self):self.reset();super().__init__()
        def reset(self):self.i=0
        def next(self,input_data):
            if self.i>=len(tr):return False
            j=tr[self.i:self.i+8192];self.i+=8192;input_data(data=cp.asarray(x[j,:width]),label=cp.asarray(y[j]));return True
    params=dict(objective='binary:logistic',tree_method='hist',device='cuda',max_depth=depth,eta=.02,subsample=.8,reg_lambda=10,seed=42,max_bin=127,colsample_bytree=.8,colsample_bynode=.4,min_child_weight=100,gamma=1,nthread=4);dm=xgb.QuantileDMatrix(Batch(),max_bin=127,nthread=4);model=xgb.train(params,dm,900);del dm
    x=np.load(CACHE/'test.npy',mmap_mode='r');va=np.arange(len(x));assert len(va)==1037090
    def pred(j):return cp.asnumpy(model.inplace_predict(cp.asarray(x[j,:width])))
    p=np.concatenate([pred(va[i:i+8192]) for i in range(0,len(va),8192)]);np.testing.assert_array_equal(p[:16],pred(va[:16][::-1])[::-1]);np.testing.assert_allclose(p[:8],np.concatenate([pred(j[None]) for j in va[:8]]),atol=1e-7);assert np.isfinite(p).all();key=f'{fold}-d{depth}-r{more}';np.save(OUT/f'{key}.npy',p);return {'fit':len(tr),'test':len(va),'width':width}


if __name__=='__main__':
    OUT.mkdir(exist_ok=True,parents=True);CACHE.mkdir(exist_ok=True,parents=True);s=sys.argv[1:];start=time.monotonic()
    if s[0]=='prepare':prepare(next(p.parent for p in Path('/kaggle/input').rglob('train.parquet') if (p.parent/'sample_submission.csv').exists()),s[1]);report={}
    else:report=fit(s[0],int(s[1]),int(s[2]))
    report['seconds']=time.monotonic()-start;(OUT/('-'.join(s)+'.json')).write_text(json.dumps(report)+'\n');print(report,flush=True)
