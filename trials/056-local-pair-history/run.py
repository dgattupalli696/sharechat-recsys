"""Historical item-pair exposures and outcomes. Run after T055."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

TRIAL=Path(__file__).resolve().parent
ROOT=TRIAL.parents[1]
OUT=ROOT/'artifacts/local-t056'
spec=importlib.util.spec_from_file_location('t055',TRIAL.parent/'055-local-rates-item-residuals/run.py'); prev=importlib.util.module_from_spec(spec); spec.loader.exec_module(prev)
HISTORY=prev.OUT


def past(key,t,values):
    """Sum values from strictly earlier timestamps within each key (zero in holdout)."""
    ix=np.lexsort((t,key)); k=key[ix]; ts=t[ix]; first=np.r_[True,k[1:]!=k[:-1]]
    change=first | np.r_[False,ts[1:]!=ts[:-1]]; start=np.maximum.accumulate(np.where(change,np.arange(len(ix)),0)); origin=np.maximum.accumulate(np.where(first,np.arange(len(ix)),0))
    out=[]
    for val in values:
        cumulative=np.r_[0,np.cumsum(val[ix],dtype=np.float64)]; result=np.empty(len(ix)); result[ix]=cumulative[start]-cumulative[origin]; out.append(result)
    return out


def features(rq,it,t,fit,res):
    # One exposure per distinct (request,item), even when it occurs on multiple rows.
    ordered=np.lexsort((it,rq)); rr,ii=rq[ordered],it[ordered]; first=np.r_[True,(rr[1:]!=rr[:-1])|(ii[1:]!=ii[:-1])]; reps=ordered[first]; inverse=np.empty(len(rq),dtype='i4'); inverse[ordered]=np.cumsum(first)-1
    finite=np.isfinite(res); cnt=np.bincount(inverse,weights=finite); r=np.bincount(inverse,weights=np.nan_to_num(res))/np.maximum(cnt,1)
    rr,ii,tt,ff=rq[reps],it[reps],t[reps],fit[reps]; ok=(cnt>0)&ff
    reqstarts=np.flatnonzero(np.r_[True,rr[1:]!=rr[:-1]]); lens=np.diff(np.r_[reqstarts,len(rr)]); reqtimes=tt[reqstarts]; assert np.array_equal(tt,np.repeat(reqtimes,lens))
    size=np.repeat(lens,lens); total=np.searchsorted(np.sort(reqtimes[ff[reqstarts]]),tt,side='left'); pop,=past(ii,tt,[ff]); maxother=np.zeros(len(rr)); seen=np.zeros(len(rr)); strongest=np.full(len(rr),-np.inf); support=np.zeros(len(rr)); signal=np.zeros(len(rr)); weight=np.zeros(len(rr))
    ends=np.repeat(reqstarts+lens,lens); seq=np.arange(len(rr),dtype='i4'); left=[];right=[]
    for delta in range(1,lens.max()):
        l=seq[seq+delta<ends];left.append(l);right.append(l+delta)
    if left:
        l=np.concatenate(left);h=np.concatenate(right); del left,right
        pair=ii[l].astype('i8')*(int(ii.max())+1)+ii[h]; stamp=tt[l]; exp,cl,ch,sl,sh=past(pair,stamp,[ff[l],ok[l],ok[h],r[l]*ok[l],r[h]*ok[h]])
        expected=pop[l]*pop[h]/np.maximum(1,total[l]); lift=exp/(exp+5)*np.log((1+exp)/(1+expected))
        for row,other,c,s in [(l,h,cl,sl),(h,l,ch,sh)]:
            np.maximum.at(maxother,row,pop[other]); np.add.at(seen,row,exp>0); np.maximum.at(strongest,row,np.where(exp>0,lift,-np.inf)); np.maximum.at(support,row,c)
            # Aggregate directional deviations, pooling by support and shrinking per partner.
            np.add.at(signal,row,np.where(c>=3,s,0));np.add.at(weight,row,np.where(c>=3,c+20,0))
    strongest[~np.isfinite(strongest)]=np.nan
    control=np.column_stack([np.log1p(pop),np.where(size>1,np.log1p(maxother),np.nan)])[inverse]
    assoc=np.column_stack([strongest,np.divide(seen,np.maximum(size-1,1),out=np.full(len(size),np.nan),where=size>1)])[inverse]
    outcome=np.divide(signal,weight,out=np.full(len(rr),np.nan),where=weight>0)[inverse]
    return control,assoc,outcome,support[inverse]


def prepare():
    f=pl.read_parquet('data/train.parquet',columns=['feature1','request_id','feature18']); v=(f['feature1']>=prev.base.CUTOFF).to_numpy(); np.testing.assert_array_equal(np.flatnonzero(v),prev.base.load('row_ids'))
    rq,it=(f[c].cast(pl.Categorical).to_physical().to_numpy() for c in ('request_id','feature18'));t=f['feature1'].cast(pl.Int64).to_numpy()
    r=prev.prepare_history(return_residual=True); baseline=np.empty(len(r));
    for mask,split in [(~v,'fit'),(v,'valid')]: baseline[mask]=np.load(HISTORY/f'history_{split}.npy').ravel()
    np.testing.assert_allclose(prev.prior(it,t,r),baseline,atol=1e-7,equal_nan=True)
    control,assoc,outcome,support=features(rq,it,t,~v,r-np.nan_to_num(baseline)); audit={}
    for mask,split in [(~v,'fit'),(v,'valid')]:
        for name,x in [('popularity',control),('association',np.column_stack([control,assoc])),('pair-outcome',np.column_stack([baseline,outcome]))]: np.save(OUT/f'{name}_{split}.npy',x[mask].astype('f4'))
        audit[split]={'rows':int(mask.sum()),'seen_pair_fraction':float(np.isfinite(assoc[mask,0]).mean()),'pair_outcome_fraction':float(np.isfinite(outcome[mask]).mean()),'max_companion_support_quantiles':np.quantile(support[mask],[0,.25,.5,.75,.9,.99,1]).tolist()}
    audit['outcome_eligible']=all(audit[p]['pair_outcome_fraction']>=.1 for p in ('fit','valid')); (TRIAL/'coverage.json').write_text(json.dumps(audit,indent=2)+'\n');print(audit,flush=True)


def train(arm):
    prev.OUT=OUT;prev.TRIAL=TRIAL;prev.train(arm)


def compare():
    y=prev.base.load('y_valid'); ids=prev.base.load('row_ids'); report={}
    for name,p in [('association',OUT/'popularity.npz'),('pair-outcome',HISTORY/'history.npz')]:
        if not (OUT/f'{name}.npz').exists():continue
        with np.load(p) as ref,np.load(OUT/f'{name}.npz') as new:
            for z in [ref,new]:np.testing.assert_array_equal(ids,z['row_ids'])
            a,b=ref['blend'],new['blend'];report[name]={'control':p.stem,'delta':roc_auc_score(y,b)-roc_auc_score(y,a),'request_bootstrap_95pct':prev.interval(y,[a,b])}
    (TRIAL/'paired.json').write_text(json.dumps(report,indent=2)+'\n');print(report,flush=True)


if __name__=='__main__':
    os.chdir(ROOT);OUT.mkdir(exist_ok=True,parents=True)
    if len(sys.argv)>1:
        if sys.argv[1]=='prepare':prepare()
        else:train(sys.argv[1])
    else:
        if not all((TRIAL.parent/'055-local-rates-item-residuals'/f'{s}.json').exists() for s in ['rates','history']):raise SystemExit('T055 fits must finish first')
        for stage,done in [('prepare','coverage'),('popularity','popularity'),('association','association'),('pair-outcome','pair-outcome')]:
            if stage=='pair-outcome' and not json.loads((TRIAL/'coverage.json').read_text())['outcome_eligible']:continue
            if not (TRIAL/f'{done}.json').exists():subprocess.run([sys.executable,'-u',__file__,stage],check=True)
        compare()
