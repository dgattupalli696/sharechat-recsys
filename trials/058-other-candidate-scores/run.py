"""Temporal second-stage experiment: what do other candidates' scores tell us?"""
import importlib.util
import json
import os
from pathlib import Path
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

trial=Path(__file__).resolve().parent; root=trial.parents[1]; os.chdir(root)
spec=importlib.util.spec_from_file_location('t055',trial.parent/'055-local-rates-item-residuals/run.py'); prev=importlib.util.module_from_spec(spec);spec.loader.exec_module(prev)


def features(request,p):
    r=np.unique(request,return_inverse=True)[1];size=np.bincount(r)[r]; z=logit(np.clip(p,1e-6,1-1e-6));n=size-1
    grouped=lambda a:np.bincount(r,weights=a)[r]-a
    mean=grouped(z)/np.maximum(n,1);sd=np.sqrt(np.maximum(0,grouped(z*z)/np.maximum(n,1)-mean**2)); total=grouped(p)
    sort=np.lexsort((-p,r));starts=np.r_[0,np.flatnonzero(np.diff(r[sort]))+1];winner=sort[starts];second=sort[np.minimum(starts+1,len(r)-1)];mx=np.where(np.arange(len(r))==winner[r],z[second[r]],z[winner[r]])
    peer=np.column_stack([mean,sd,total,mx,z-mx, -np.expm1(grouped(np.log1p(-np.clip(p,0,1-1e-6))))]);peer[n==0]=np.nan
    return np.column_stack([z,size,peer])


def main():
    start=time.monotonic(); base=prev.base; f=pl.read_parquet('data/train.parquet',columns=['feature1','request_id','watched']);v=(f['feature1']>=base.CUTOFF).to_numpy();req=f['request_id'].to_numpy(); t=f['feature1'].cast(pl.Int64).to_numpy(); y=base.load('y_valid');ids=base.load('row_ids');np.testing.assert_array_equal(np.flatnonzero(v),ids)
    inner=np.load('artifacts/local-t057/inner-pooled.npz'); ix=inner['ix']; fitrows=np.flatnonzero(~v)[ix];assert t[fitrows].min()>t[np.setdiff1d(np.flatnonzero(~v),fitrows)].max();assert t[fitrows].max()<t[v].min(); yy=base.load('y_fit')[ix]; np.testing.assert_array_equal(yy,f['watched'].to_numpy()[fitrows]);np.testing.assert_array_equal(y,f['watched'].to_numpy()[ids]);z=np.load(base.OUT/'A.npz');np.testing.assert_array_equal(z['row_ids'],ids);peers=base.peerscore();a=peers+rankdata(z['probabilities']);np.testing.assert_array_equal(a,z['blend'])
    fx=features(req[fitrows],inner['p'].astype('d'));vx=features(req[ids],z['probabilities'].astype('d'))
    assert set(req[fitrows]).isdisjoint(req[ids]);params=dict(objective='binary',learning_rate=.04,num_leaves=15,min_data_in_leaf=300,lambda_l2=10,boost_from_average=False,num_threads=4,deterministic=True,force_col_wise=True,verbosity=-1,seed=42); report={'baseline_auc':roc_auc_score(y,a),'meta_train_rows':len(ix),'eval_rows':len(ids)};blends=[]; groups=np.unique(req[ids],return_inverse=True)[1]
    for name,dim in [('own_score_and_size',2),('other_candidate_scores',vx.shape[1])]:
        m=lgb.train(params,lgb.Dataset(fx[:,:dim],label=yy,init_score=fx[:,0]),num_boost_round=160);pred=vx[:,0]+m.predict(vx[:,:dim],raw_score=True);b=peers+rankdata(pred);blends.append(b);auc=roc_auc_score(y,b);report[name]={'blend_auc':auc,'delta_vs_A':auc-report['baseline_auc'],'delta_95pct':prev.interval(y,[a,b]),'standalone_auc':roc_auc_score(y,pred)}
    report['peer_vs_own_delta']=report['other_candidate_scores']['blend_auc']-report['own_score_and_size']['blend_auc'];report['peer_vs_own_95pct']=prev.interval(y,blends);report['seconds']=time.monotonic()-start;(trial/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
