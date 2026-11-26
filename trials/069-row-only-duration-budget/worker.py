"""T069 row-only duration interactions x training budget, fixed rich depth-6 backbone."""
import sys,time,json,gc
from pathlib import Path
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import base62 as base
from rich63 import rich

OUT=Path('/kaggle/working/t069'); CACHE=Path('/kaggle/temp/t069'); H=[476,477,479,480,483,484,486,487,492,493,500,501,502,503,512,513,514,515];BASE_WIDTH=1105

def duration(f):
    d=f['feature54'].to_numpy().astype('d');s=np.column_stack([f[f'feature{i}'].to_numpy() for i in H]).astype('d');d=np.where(d>0,d,np.nan);s=np.where(s>=0,s,np.nan)
    return np.column_stack([np.log1p(s)-np.log1p(d[:,None]),s-d[:,None]]).astype('f4')

def prepare(raw):
    names=base.numeric(pl.scan_parquet(raw).collect_schema());fields=names+[f'feature{i}' for i in [*range(26,33),*range(40,47)]];assert len(names)==584 and 'request_id' not in fields
    pf=pq.ParquetFile(raw);assert pf.metadata.num_rows==1537638;x=np.lib.format.open_memmap(CACHE/'x.npy',mode='w+',dtype='f4',shape=(1537638,BASE_WIDTH+2*len(H)));off=0
    for batch in pf.iter_batches(columns=fields,batch_size=8192):
        f=pl.from_arrow(batch);x[off:off+len(f)]=np.column_stack([base.transform(f,names),rich(f),duration(f)]);off+=len(f)
    assert off==len(x);x.flush();f=pl.read_parquet(raw,columns=['feature1','watched']);np.save(CACHE/'t.npy',f['feature1'].to_numpy());np.save(CACHE/'y.npy',f['watched'].to_numpy())

def fit(fold,extra):
    import cupy as cp
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score
    t=np.load(CACHE/'t.npy');y=np.load(CACHE/'y.npy');x=np.load(CACHE/'x.npy',mmap_mode='r');cut=base.CUTOFF if fold=='primary' else np.datetime64('2026-10-27T08:06:52.397');tr=np.flatnonzero(t<cut);va=np.flatnonzero((t>=cut)&((fold=='primary')|(t<np.datetime64('2026-10-27T18:00:00'))));assert (len(tr),len(va))==((1230110,307528) if fold=='primary' else (710330,299969));width=x.shape[1] if extra else BASE_WIDTH
    class Iter(xgb.DataIter):
        def __init__(self):self.reset();super().__init__()
        def reset(self):self.i=0
        def next(self,input_data):
            if self.i>=len(tr):return False
            j=tr[self.i:self.i+8192];self.i+=8192;input_data(data=cp.asarray(x[j,:width]),label=cp.asarray(y[j]));return True
    params=dict(objective='binary:logistic',tree_method='hist',device='cuda',max_depth=6,eta=.02,subsample=.8,reg_lambda=10,seed=42,max_bin=127,colsample_bytree=.8,colsample_bynode=.4,min_child_weight=100,gamma=1,nthread=4);dm=xgb.QuantileDMatrix(Iter(),max_bin=127,nthread=4);model=xgb.train(params,dm,2400);del dm;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    for rounds in [900,1600,2400]:
        def pred(j):return cp.asnumpy(model.inplace_predict(cp.asarray(x[j,:width]),iteration_range=(0,rounds)))
        p=np.concatenate([pred(va[i:i+8192]) for i in range(0,len(va),8192)]);np.testing.assert_allclose(p[:8],np.concatenate([pred(j[None]) for j in va[:8]]),atol=1e-7);assert np.isfinite(p).all();key=f'{fold}-{extra}-{rounds}';np.savez_compressed(OUT/f'{key}.npz',row_ids=va,y_true=y[va],probabilities=p);print(key,roc_auc_score(y[va],p),flush=True)

if __name__=='__main__':
    OUT.mkdir(exist_ok=True,parents=True);CACHE.mkdir(exist_ok=True,parents=True);args=sys.argv[1:];start=time.monotonic()
    if args==['prepare']:prepare(next(p for p in Path('/kaggle/input').rglob('train.parquet') if (p.parent/'sample_submission.csv').is_file()))
    else:fit(args[0],int(args[1]))
    (OUT/('-'.join(args)+'.json')).write_text(json.dumps({'seconds':time.monotonic()-start})+'\n')
