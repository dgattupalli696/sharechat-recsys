"""Regularized joint probability weights: learn early, select shrinkage later, transfer to primary."""
import json,runpy
from pathlib import Path
import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score
t=Path(__file__).resolve().parent;root=t.parents[1];art=root/'artifacts';m=runpy.run_path(str(t.parent/'065-row-only-blend-weights/run.py'));ts=pl.read_parquet(root/'data/train.parquet',columns=['feature1'])['feature1'].to_numpy();w0=np.r_[np.r_[.5*.83125,np.repeat(.125*.83125,4),.11875,.05]*.833,.167];report={}
def pool(q,z,w):return .25*(q@w)+.75*expit(z@w)
for fold in ['earlier','primary']:
    ids,y,anchor,p=m['fold'](fold)
    def load(path):
        i,l,q=m['read'](path);np.testing.assert_array_equal(i,ids);np.testing.assert_array_equal(l,y);return q
    q=np.column_stack([anchor,*[p[k] for k in m['trees']],load(art/f'kaggle-t069/t069/{fold}-1-2400.npz'),load(art/f'kaggle-t070/t070/{fold}-gated-8.npz'),load(art/f'kaggle-t072/t072/{fold}-xgb-base.npz')]);z=logit(np.clip(q,1e-8,1-1e-8));base=pool(q,z,w0);auc=roc_auc_score(y,base);assert abs(auc-{'primary':.84223384,'earlier':.83873085}[fold])<1e-7
    if fold=='earlier':
        fit=ts[ids]<np.sort(ts[ids])[len(ids)//2];rng=np.random.default_rng(42);pos=rng.choice(np.flatnonzero(fit&(y==1)),250000);neg=rng.choice(np.flatnonzero(fit&(y==0)),250000);d=z[pos]-z[neg];options=[w0];
        # Smooth AUC surrogate with temperature .25, averaged over fixed positive-negative pairs.
        for reg in [.0001,.001,.01]:
            def fun(w):
                s=d@w/.25;return np.logaddexp(0,-s).mean()+reg*np.sum((w-w0)**2),-(d.T@expit(-s))/(len(s)*.25)+2*reg*(w-w0)
            r=minimize(fun,w0,jac=True,method='SLSQP',bounds=[(0,1)]*len(w0),constraints=[dict(type='eq',fun=lambda w:w.sum()-1)],options=dict(ftol=1e-12,maxiter=100));assert r.success,r.message
            options.extend([w0+f*(r.x-w0) for f in [.25,.5,1]])
        select=[roc_auc_score(y[~fit],pool(q[~fit],z[~fit],w)) for w in options];winner=options[int(np.argmax(select))];report['selection']=dict(weights=winner.tolist(),option=int(np.argmax(select)),validation_gain=max(select)-select[0])
    s=pool(q,z,winner);aa=roc_auc_score(y,s);report[fold]=dict(auc=aa,delta_vs_T078=aa-auc,delta_vs_S023=aa-roc_auc_score(y,.5*(anchor+p['all4'])))
report['promote']=report['primary']['delta_vs_S023']>=.0003 and all(report[f]['delta_vs_T078']>0 for f in ['earlier','primary']);(t/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
