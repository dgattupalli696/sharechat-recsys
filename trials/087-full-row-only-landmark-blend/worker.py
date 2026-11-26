"""All-label refit of the components required by the frozen T086 selection."""
from pathlib import Path
import sys,json,time
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import base62,rich63,t069,rot83,land86

N=1537638;V=1037090;OUT=Path('/kaggle/working/t087');CACHE=Path('/kaggle/temp/t087')
def matrix(f,names):return np.column_stack([base62.transform(f,names),rich63.rich(f),t069.duration(f)])
def compact(z):return np.column_stack([z[:,:594],z[:,1105:],z[:,1042:1091]])
def prepare(data):
    names=base62.numeric(pl.scan_parquet(data/'train.parquet').collect_schema());cols=names+[f'feature{i}' for i in [*range(26,33),*range(40,47)]];assert 'request_id' not in cols
    x=np.lib.format.open_memmap(CACHE/'x.npy',mode='w+',dtype='f4',shape=(N+V,1141));off=0
    for name,rows in [('train',N),('test',V)]:
        raw=pq.ParquetFile(data/f'{name}.parquet');assert raw.metadata.num_rows==rows
        for batch in raw.iter_batches(columns=cols,batch_size=8192):
            f=pl.from_arrow(batch);z=matrix(f,names);x[off:off+len(f)]=z;off+=len(f)
    assert off==len(x);x.flush();np.save(CACHE/'y.npy',pl.read_parquet(data/'train.parquet',columns=['watched'])['watched'].to_numpy());ids=pl.read_csv(data/'sample_submission.csv')['id'];assert ids.n_unique()==len(ids)==V;np.save(OUT/'ids.npy',ids.to_numpy())

def tree(arm):
    import cupy as cp
    import xgboost as xgb
    x=np.load(CACHE/'x.npy',mmap_mode='r');vec=x[:,594:1042].reshape(len(x),14,32);tr=np.arange(N);y=np.load(CACHE/'y.npy');assert len(y)==N
    mod=rot83 if arm=='paired' else land86 if arm in ('learned','random') else None
    state=mod.learn(vec,tr,False if arm=='paired' else arm) if mod else None
    def feat(i,j):
        z=x[i:j];v=z if arm=='duration' else compact(z);return np.ascontiguousarray(np.column_stack([v,mod.apply(vec[i:j],state)]) if mod else v)
    class Batch(xgb.DataIter):
        def __init__(self):self.reset();super().__init__()
        def reset(self):self.i=0
        def next(self,input_data):
            if self.i>=N:return False
            i=self.i;j=min(N,i+8192);self.i=j;input_data(data=cp.asarray(feat(i,j)),label=cp.asarray(y[i:j]));return True
    dur=arm=='duration';params=dict(objective='binary:logistic',tree_method='hist',device='cuda',max_depth=6 if dur else 8,eta=.02 if dur else .025,subsample=.8,reg_lambda=10,seed=42,max_bin=127,colsample_bytree=.8 if dur else .65,colsample_bynode=.4 if dur else .5,min_child_weight=100,gamma=1,nthread=4)
    dm=xgb.QuantileDMatrix(Batch(),max_bin=127,nthread=4);model=xgb.train(params,dm,2400 if dur else 1600);del dm
    def pred(i,j):return cp.asnumpy(model.inplace_predict(cp.asarray(feat(i,j))))
    p=np.concatenate([pred(i,min(len(x),i+8192)) for i in range(N,len(x),8192)]);np.testing.assert_allclose(p[:8],np.concatenate([pred(i,i+1) for i in range(N,N+8)]),atol=1e-6);save(arm,p)

def save(arm,p):
    assert p.shape==(V,) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all();np.save(OUT/f'{arm}.npy',p)

if __name__=='__main__':
    OUT.mkdir(exist_ok=True,parents=True);CACHE.mkdir(exist_ok=True,parents=True);stage=sys.argv[1];start=time.monotonic()
    if stage=='prepare':prepare(next(p.parent for p in Path('/kaggle/input').rglob('train.parquet') if (p.parent/'sample_submission.csv').exists()))
    elif stage in ('gated','quantile'):
        import neural87
        neural87.fit(stage)
    else:tree(stage)
    (OUT/f'{stage}.json').write_text(json.dumps(dict(stage=stage,fit=N,test=V,seconds=time.monotonic()-start))+'\n');print('completed',stage,flush=True)
