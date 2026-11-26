"""Row-independent XGB with raw numeric features and within-row embedding cosines."""
import json
from pathlib import Path
import sys
import time

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
import xgboost as xgb

TRIAL=Path(__file__).resolve().parent; ROOT=TRIAL.parents[1];sys.path.insert(0,str(ROOT))
from embed_screen import COS_PAIRS,unit_vectors
from feature_screen import CUTOFF,scalar_numeric
from xgb_tune import BASE,CONFIGS


def transform(f,names):
    x=np.empty((len(f),len(names)+len(COS_PAIRS)),np.float32)
    for i,n in enumerate(names):x[:,i]=f[n].to_numpy()
    for j,(u,v) in enumerate(COS_PAIRS): x[:,len(names)+j]=(unit_vectors(f[u])*unit_vectors(f[v])).sum(1)
    return x


def main():
    start=time.monotonic(); raw=ROOT/'data/train.parquet';schema=pl.scan_parquet(raw).collect_schema();names=list(dict.fromkeys(['feature3',*scalar_numeric(schema)])); assert all(n.startswith('feature') and schema[n].is_numeric() for n in names)
    fields=names+list(dict.fromkeys(c for pair in COS_PAIRS for c in pair)); out=ROOT/'artifacts/local-t059';out.mkdir(exist_ok=True,parents=True)
    # Construct disk-backed matrices in bounded batches. Never read request ID during feature extraction.
    import pyarrow.parquet as pq
    meta=pl.read_parquet(raw,columns=['feature1','watched']);valid=(meta['feature1']>=CUTOFF).to_numpy();y=meta['watched'].to_numpy();ids=np.flatnonzero(valid);w=len(names)+len(COS_PAIRS)
    train=np.lib.format.open_memmap(out/'fit.npy',mode='w+',dtype='f4',shape=(int((~valid).sum()),w)); val=np.lib.format.open_memmap(out/'valid.npy',mode='w+',dtype='f4',shape=(int(valid.sum()),w));offset=ft=vt=0;probe=None
    for batch in pq.ParquetFile(raw).iter_batches(columns=fields,batch_size=8192):
        frame=pl.from_arrow(batch); x=transform(frame,names);mask=valid[offset:offset+len(x)];a,b=(~mask).sum(),mask.sum();train[ft:ft+a]=x[~mask];val[vt:vt+b]=x[mask];ft+=a;vt+=b;offset+=len(x)
        if probe is None and b>=20:probe=frame.filter(pl.Series(mask)).head(20)
    assert (ft,vt,offset)==(len(train),len(val),len(meta));assert probe is not None
    class Batch(xgb.DataIter):
        def __init__(self,x,labels):self.x=x;self.y=labels;self.reset();super().__init__()
        def reset(self):self.pos=0
        def next(self,input_data):
            if self.pos>=len(self.x):return False
            s=slice(self.pos,self.pos+8192);self.pos+=8192;input_data(data=self.x[s],label=self.y[s]);return True
    fit=xgb.QuantileDMatrix(Batch(train,y[~valid]),max_bin=127,nthread=6);test=xgb.QuantileDMatrix(Batch(val,y[valid]),max_bin=127,nthread=6,ref=fit)
    params={**BASE,**CONFIGS['depth12_node'],'device':'cpu','nthread':6};model=xgb.train(params,fit,900,evals=[(test,'holdout')],verbose_eval=100);p=model.predict(test); model.save_model(out/'model.ubj')
    # Exercise raw feature construction AND scoring with different batch populations.
    batch_pred=lambda frame:model.inplace_predict(transform(frame,names));np.testing.assert_array_equal(batch_pred(probe),batch_pred(probe.reverse())[::-1]);np.testing.assert_array_equal(batch_pred(probe),np.concatenate([batch_pred(probe.slice(i,1)) for i in range(len(probe))]))
    with np.load(ROOT/'artifacts/local-t016/validation_predictions.npz') as z:
        order=np.argsort(z['row_ids']);pos=order[np.searchsorted(z['row_ids'][order],ids)];np.testing.assert_array_equal(z['row_ids'][pos],ids);np.testing.assert_array_equal(z['y_true'].ravel()[pos],y[valid]);ref=z['probabilities'].ravel()[pos]
    blend=.75*ref+.25*p;baseline=roc_auc_score(y[valid],ref);assert abs(baseline-.8408057533)<1e-8;score=roc_auc_score(y[valid],blend)
    report=dict(train_rows=int(ft),validation_rows=int(vt),features=w,standalone_auc=roc_auc_score(y[valid],p),T016_auc=baseline,fixed_blend_auc=score,delta=score-baseline,batch_invariance_probe_rows=len(probe),seconds=time.monotonic()-start);np.savez_compressed(out/'predictions.npz',row_ids=ids,probabilities=p,blend=blend);(TRIAL/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)


if __name__=='__main__':main()
