"""Local specialist and matched-viewer tests. Execute from the repository root."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import polars as pl
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb

TRIAL=Path(__file__).resolve().parent
ROOT=TRIAL.parents[1]; OUT=ROOT/'artifacts/local-t057'
spec=importlib.util.spec_from_file_location('t055',TRIAL.parent/'055-local-rates-item-residuals/run.py'); prev=importlib.util.module_from_spec(spec); spec.loader.exec_module(prev)
base=prev.base; names=json.loads((base.M/'columns.json').read_text())['columns']; rounds=941


def write(name,obj):
    (TRIAL/f'{name}.json').write_text(json.dumps(obj,indent=2)+'\n');print(name,obj,flush=True)


def meta():
    f=pl.read_parquet('data/train.parquet',columns=['feature1','request_id','feature18','feature5','watched']); v=(f['feature1']>=base.CUTOFF).to_numpy()
    np.testing.assert_array_equal(np.flatnonzero(v),base.load('row_ids'))
    for mask,s in [(~v,'fit'),(v,'valid')]: np.testing.assert_array_equal(f['watched'].to_numpy()[mask],base.load('y_'+s))
    return f,v


class Rows(xgb.DataIter):
    def __init__(self, split, ix):
        self.x=base.load('x_'+split); self.y=base.load('y_'+split); self.ix=ix;self.reset();super().__init__()
    def reset(self):self.i=0
    def next(self,input_data):
        if self.i>=len(self.ix):return False
        j=self.ix[self.i:self.i+8192];self.i+=8192;input_data(data=self.x[j],label=self.y[j]);return True


def short_fit(stage):
    f,v=meta(); t=f['feature1'].cast(pl.Int64).to_numpy()[~v]; del f
    edge=np.quantile(t,.75,method='inverted_cdf'); early=t<edge; n=base.load('x_fit')[:,names.index('rq_size')]; nv=base.load('x_valid')[:,names.index('rq_size')]; assert max(t[early])<min(t[~early])
    train=np.flatnonzero((early if stage!='full-short' else np.ones(len(t),bool)) & ((n<=3) if stage!='inner-pooled' else True)); split='valid' if stage=='full-short' else 'fit'
    ix=np.flatnonzero(nv<=3) if stage=='full-short' else np.flatnonzero(~early & ((n<=3) if stage=='inner-short' else True)); began=time.monotonic()
    # Conditional fits use half the pooled budget (smaller dataset); frozen before evaluation.
    iters=rounds if stage=='inner-pooled' else 470
    d=xgb.QuantileDMatrix(Rows('fit',train),max_bin=127,nthread=6); q=xgb.QuantileDMatrix(Rows(split,ix),max_bin=127,nthread=6,ref=d)
    model=xgb.train({**base.BASE,**base.CONFIGS['depth12_node'],'device':'cpu','nthread':6},d,iters)
    pred=model.predict(q);np.savez_compressed(OUT/f'{stage}.npz',ix=ix,p=pred);print(stage,'fit',len(train),'score',len(ix),'rounds',iters,'seconds',time.monotonic()-began,flush=True)


def score_arms():
    pooled=np.load(OUT/'inner-pooled.npz'); special=np.load(OUT/'inner-short.npz'); ix=pooled['ix']; size=base.load('x_fit')[ix,names.index('rq_size')]; y=base.load('y_fit')[ix]; small=size<=3
    np.testing.assert_array_equal(ix[small],special['ix']); p=pooled['p']; nv=base.load('x_valid')[:,names.index('rq_size')]; short=nv<=3
    z=np.load(base.OUT/'A.npz'); ref=z['blend']; outer=z['probabilities']; np.testing.assert_array_equal(base.load('row_ids'),z['row_ids']); full=np.load(OUT/'full-short.npz');np.testing.assert_array_equal(full['ix'],np.flatnonzero(short))
    def lg(values):return logit(np.clip(values.astype('d'),1e-6,1-1e-6))
    # Calibrate both heads and the long-request anchor on the first half of the inner validation period.
    f,v=meta(); t=f['feature1'].cast(pl.Int64).to_numpy()[~v][ix]; del f
    tune=t>=np.quantile(t,.5,method='inverted_cdf'); assert max(t[~tune])<min(t[tune]);raw=lg(p); test=lg(outer); patch=raw.copy();outer_patch=test.copy(); sp=np.full(len(p),np.nan);sp[small]=lg(special['p']);so=np.full(len(outer),np.nan);so[short]=lg(full['p'])
    def calibrator(mask,arr):
        keep=mask & ~tune; m=LogisticRegression(C=1,max_iter=150).fit(arr[keep,None],y[keep]);return m.coef_.item(),m.intercept_.item()
    slope,offset=calibrator(~small,raw); assert slope>0
    for group in [1,2]:
        g=(size==1) if group==1 else (small & (size>1)); h=(nv==1) if group==1 else (short & (nv>1))
        for arr,out,source,osource in [(raw,test,lg(p),lg(outer)),(patch,outer_patch,sp,so)]:
            a,b=calibrator(g,source); arr[g]=(a*source[g]+b-offset)/slope;out[h]=(a*osource[h]+b-offset)/slope
    options=[0,.25,.5,.75,1.]; vals=[float(roc_auc_score(y[tune],((1-w)*raw+w*patch)[tune])) for w in options]; weight=options[np.argmax(vals)]
    peers=base.peerscore(); yy=base.load('y_valid'); np.testing.assert_array_equal(ref,peers+rankdata(outer)); blends=[]; report={'inner_auc_by_weight':dict(zip(options,vals)),'weight':weight,'baseline_auc':roc_auc_score(yy,ref)}
    for s,logits in [('size-calibration',test),('specialist',(1-weight)*test+weight*outer_patch)]:
        np.testing.assert_allclose(logits[~short],lg(outer[~short])); pred=expit(logits); blend=peers+rankdata(pred); blends.append(blend); auc=roc_auc_score(yy,blend)
        np.savez_compressed(OUT/f'{s}.npz',row_ids=base.load('row_ids'),probabilities=pred,blend=blend)
        report[s]={'blend_auc':auc,'delta_vs_A':auc-report['baseline_auc'],'paired_95pct':prev.interval(yy,[ref,blend]),'short_rows_auc':roc_auc_score(yy[short],blend[short])}
    report['specialist_minus_calibration']=report['specialist']['blend_auc']-report['size-calibration']['blend_auc'];report['specialist_minus_calibration_95pct']=prev.interval(yy,blends);write('specialist',report)


def make_pairs(ix,key,rq,user,y,rng):
    # Up to three controls per watched row; same item and size bucket, distinct viewer and request.
    neg=ix[y[ix]==0]; neg=neg[np.argsort(key[neg],kind='stable')]; pos=ix[y[ix]==1]; lo=np.searchsorted(key[neg],key[pos]);hi=np.searchsorted(key[neg],key[pos],side='right'); has=hi>lo; pos,lo,hi=pos[has],lo[has],hi[has]; p=np.repeat(pos,3);q=neg[rng.integers(np.repeat(lo,3),np.repeat(hi,3))]; ok=(rq[p]!=rq[q])&(user[p]!=user[q]);pq=np.unique(np.column_stack([p[ok],q[ok]]),axis=0);return pq[:,0],pq[:,1]


def prepare_audience():
    f,v=meta(); xfit=base.load('x_fit');xvalid=base.load('x_valid'); y=base.load('y_fit');size=xfit[:,names.index('rq_size')];t=f['feature1'].cast(pl.Int64).to_numpy()[~v];codes=lambda c: f[c].filter(pl.Series(~v)).cast(pl.Categorical).to_physical().to_numpy().astype('i8'); item,rq,user=map(codes,['feature18','request_id','feature5']);key=item*4+np.digitize(size,[1.5,3.5,6.5]);rng=np.random.default_rng(42)
    columns=[f'feature{i}' for i in [3,33,34,35,36,37,38,39,54,60,66,67,68,69,70,71,72,416,417,418,419,430,431,475,476,477,478,479,480,483,484,492,493]]+['rq_size']+[s for s in names if s.startswith('cos_') or s.startswith('rq_rank_cos_')]+[f'rq_rank_feature{i}' for i in (410,411,416,417,418,419,430,431) if f'rq_rank_feature{i}' in names]; idx=[names.index(c) for c in columns]
    x=np.ascontiguousarray(xfit[:,idx]); xv=np.ascontiguousarray(xvalid[:,idx]);edges=np.quantile(t,[.2,.4,.6,.8],method='inverted_cdf'); block=np.searchsorted(edges,t,side='right');scores=np.full(len(y),np.nan,dtype='f4');audits=[]
    for k in range(1,6):
        tr=np.flatnonzero(block<k);p,q=make_pairs(tr,key,rq,user,y,rng);assert len(p)>1000 and np.all(key[p]==key[q]) and np.all(user[p]!=user[q]) and np.all(rq[p]!=rq[q]); subset=rng.choice(tr,min(30000,len(tr)),replace=False);sample=x[subset];lo,hi=np.nanpercentile(sample,[1,99],axis=0);med=np.nanmedian(sample,axis=0);scale=np.maximum(np.nanstd(np.clip(sample,lo,hi),axis=0),1e-5)
        def phi(a):
            a=np.nan_to_num((np.clip(a,lo,hi)-med)/scale).astype('f4'); duration=a[:,columns.index('feature54'):columns.index('feature54')+1];return np.column_stack([a,a[:,1:8]*duration])
        delta=phi(x[p])-phi(x[q]);model=LogisticRegression(C=.1,max_iter=200,fit_intercept=False).fit(np.concatenate([delta,-delta]),np.repeat([1,0],len(p)));loc=model.decision_function(phi(sample));center,spread=loc.mean(),max(loc.std(),1e-6)
        def predict(a):return (model.decision_function(phi(a))-center)/spread
        if k<5:
            rows=np.flatnonzero(block==k);assert t[tr].max()<t[rows].min(); scores[rows]=predict(x[rows]);ep,eq=make_pairs(rows,key,rq,user,y,rng);acc=float((scores[ep]>scores[eq]).mean()) if len(ep) else None
        else:outer=predict(xv).astype('f4');acc=None
        audit={'block':k,'pairs':len(p),'items':int(np.unique(item[p]).size),'positive_rows':int(np.unique(p).size),'next_block_pair_win_fraction':acc};audits.append(audit);print(audit,flush=True)
    assert np.isfinite(outer).all() and np.array_equal(np.isfinite(scores),block>0)
    for split,s in [('fit',scores),('valid',outer)]:np.save(OUT/f'audience_{split}.npy',s[:,None])
    write('audience-audit',{'columns':columns,'folds':audits,'train_score_coverage':float(np.isfinite(scores).mean())})


if __name__=='__main__':
    os.chdir(ROOT);OUT.mkdir(exist_ok=True,parents=True)
    if len(sys.argv)>1:
        stage=sys.argv[1]
        if stage in ['inner-pooled','inner-short','full-short']:short_fit(stage)
        elif stage=='specialist':score_arms()
        elif stage=='prepare-audience':prepare_audience()
        elif stage=='audience':prev.OUT=OUT;prev.TRIAL=TRIAL;prev.train('audience')
        else:raise ValueError(stage)
    else:
        for stage,done in [('inner-pooled',OUT/'inner-pooled.npz'),('inner-short',OUT/'inner-short.npz'),('full-short',OUT/'full-short.npz'),('specialist',TRIAL/'specialist.json'),('prepare-audience',TRIAL/'audience-audit.json'),('audience',TRIAL/'audience.json')]:
            if not done.exists():print('starting',stage,flush=True);subprocess.run([sys.executable,'-u',__file__,stage],check=True)
