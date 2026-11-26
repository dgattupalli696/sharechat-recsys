"""Private Kaggle workers. Row-local transform shared by both GBDT families."""
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import polars as pl
import pyarrow.parquet as pq

CUTOFF=np.datetime64('2026-10-28T08:06:52.397'); PAIRS=list(zip(range(26,33),range(40,47)))+[(32,43),(30,46),(31,44)]
CATS=[5,6,7,8,9,10,11,12,13,14,16,17,18,19,20,21,22,23,25];OUT=Path('/kaggle/working/t062');CACHE=Path('/kaggle/temp/t062')


def numeric(schema):return ['feature3']+[c for c,t in schema.items() if c.startswith('feature') and c not in ('feature1','feature3') and str(t) in ('Float32','Int64','Int8')]


def transform(f,names):
    x=np.empty((len(f),len(names)+len(PAIRS)),np.float32)
    for j,c in enumerate(names):x[:,j]=f[c].to_numpy()
    def unit(k):
        a=f[f'feature{k}'].fill_null([0.]*32).list.to_array(32).to_numpy().astype('f4');return a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-6)
    for j,(u,v) in enumerate(PAIRS):x[:,len(names)+j]=(unit(u)*unit(v)).sum(1)
    return x


def prepare(data):
    names=numeric(pl.scan_parquet(data/'train.parquet').collect_schema());fields=names+list(dict.fromkeys(f'feature{k}' for pair in PAIRS for k in pair));assert len(names)+len(PAIRS)==594 and 'request_id' not in fields
    for split,n in [('train',1537638),('test',1037090)]:
        raw=pq.ParquetFile(data/f'{split}.parquet');assert raw.metadata.num_rows==n
        x=np.lib.format.open_memmap(CACHE/f'{split}.npy',mode='w+',dtype='f4',shape=(n,594));off=0
        for batch in raw.iter_batches(columns=fields,batch_size=8192):
            f=pl.from_arrow(batch);rows=transform(f,names);x[off:off+len(f)]=rows
            if off==0:np.testing.assert_array_equal(rows[:8],np.vstack([transform(f.slice(i,1),names) for i in range(8)]))
            off+=len(f)
        assert off==n;x.flush();del x
    meta=pl.read_parquet(data/'train.parquet',columns=['feature1','watched']);hold=meta['feature1'].to_numpy()>=CUTOFF;assert hold.sum()==307528
    np.save(CACHE/'hold.npy',hold);np.save(CACHE/'y.npy',meta['watched'].to_numpy());print('prepared 594 row-only fields',flush=True)


def save_valid(name,ix,y,p):
    from sklearn.metrics import roc_auc_score
    assert np.isfinite(p).all() and np.all((p>=0)&(p<=1));np.savez_compressed(OUT/f'{name}-holdout.npz',row_ids=ix,y_true=y[ix],probabilities=p);print(name,'holdout',roc_auc_score(y[ix],p),flush=True)


def xgb_run(data,name,full):
    import cupy as cp
    import xgboost as xgb
    x=np.load(CACHE/'train.npy',mmap_mode='r');y=np.load(CACHE/'y.npy');h=np.load(CACHE/'hold.npy');ix=np.arange(len(y)) if full else np.flatnonzero(~h); val=np.flatnonzero(h)
    class Iterator(xgb.DataIter):
        def __init__(self):self.reset();super().__init__()
        def reset(self):self.pos=0
        def next(self,input_data):
            if self.pos>=len(ix):return False
            rows=ix[self.pos:self.pos+8192];self.pos+=8192;input_data(data=cp.asarray(x[rows]),label=cp.asarray(y[rows]));return True
    params=dict(objective='binary:logistic',eval_metric='auc',tree_method='hist',device='cuda',eta=.02,subsample=.8,reg_lambda=10,seed=42,max_bin=127,nthread=4)
    if name=='depth12':params.update(max_depth=12,max_leaves=0,grow_policy='depthwise',colsample_bytree=.8,colsample_bynode=.4,min_child_weight=100,gamma=1)
    else:params.update(max_depth=0,max_leaves=511,grow_policy='lossguide',colsample_bytree=.3,min_child_weight=200,reg_alpha=1)
    fit=xgb.QuantileDMatrix(Iterator(),max_bin=127,nthread=4);model=xgb.train(params,fit,900);del fit;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    arr=np.load(CACHE/'test.npy',mmap_mode='r') if full else x; ids=np.arange(len(arr)) if full else val
    def predict(rows):return cp.asnumpy(model.inplace_predict(cp.asarray(arr[rows])))
    p=np.concatenate([predict(ids[s:s+8192]) for s in range(0,len(ids),8192)]);np.testing.assert_allclose(p[:8],np.concatenate([predict(ids[j:j+1]) for j in range(8)]),atol=1e-7,rtol=0)
    if full:
        assert len(ix)==1537638 and np.isfinite(p).all() and np.all((0<=p)&(p<=1));model.save_model(OUT/'depth12-full.ubj');sample=pl.read_csv(data/'sample_submission.csv');assert len(sample)==len(p) and sample['id'].n_unique()==len(p);sample.with_columns(pl.Series('watched',p)).write_csv(OUT/'submission-full-depth12.csv');print('full fit rows',len(ix),'test rows',len(p),flush=True)
    else:save_valid(name,val,y,p)


def cat_run(data):
    import pandas as pd
    from catboost import CatBoostClassifier,Pool
    x=np.load(CACHE/'train.npy',mmap_mode='r');y=np.load(CACHE/'y.npy');h=np.load(CACHE/'hold.npy');names=[f'feature{k}' for k in CATS];cats=pl.read_parquet(data/'train.parquet',columns=names).to_pandas()
    def pool(ix):
        f=pd.DataFrame(x[ix],columns=[f'n{k}' for k in range(x.shape[1])]);
        for c in names:f[c]=cats[c].iloc[ix].fillna('__NA__').to_numpy()
        return Pool(f,label=y[ix],cat_features=names,thread_count=4)
    tr=np.flatnonzero(~h);va=np.flatnonzero(h);fit=pool(tr);valid=pool(va);model=CatBoostClassifier(iterations=3000,depth=8,learning_rate=.06,l2_leaf_reg=10,border_count=128,eval_metric='AUC',task_type='GPU',devices='0',gpu_ram_part=.85,max_ctr_complexity=2,random_seed=42,od_type='Iter',od_wait=300,verbose=250);model.fit(fit,eval_set=valid,use_best_model=True);p=model.predict_proba(valid)[:,1];np.testing.assert_allclose(model.predict_proba(pool(va[:1]))[:,1],p[:1],atol=1e-7);save_valid('catboost-ids',va,y,p);print('best trees',model.tree_count_,flush=True)


if __name__=='__main__':
    stage=sys.argv[1];OUT.mkdir(exist_ok=True,parents=True);CACHE.mkdir(exist_ok=True,parents=True);data=next(p.parent for p in Path('/kaggle/input').rglob('train.parquet') if (p.parent/'sample_submission.csv').is_file());t=time.monotonic()
    if stage=='prepare':prepare(data)
    elif stage=='catboost-ids':cat_run(data)
    else:xgb_run(data,stage.split('-')[0],stage.endswith('-full'))
    (OUT/f'{stage}.json').write_text(json.dumps({'stage':stage,'seconds':time.monotonic()-t,'full_train_rows':1537638 if stage.endswith('-full') else None})+'\n')
