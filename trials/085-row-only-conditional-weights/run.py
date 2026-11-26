"""Predeclared pointwise routing tests for T083; earlier selection, primary transfer."""
from pathlib import Path
import json,sys,runpy
import numpy as np, polars as pl
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score
t=Path(__file__).resolve().parent;root=t.parents[1];a=root/'artifacts';m=runpy.run_path(str(t.parent/'065-row-only-blend-weights/run.py'));result={};attrs=pl.scan_parquet(root/'data/train.parquet').select('feature54',pl.col('feature32').is_not_null().alias('present')).collect()
for fold in ['earlier','primary']:
    ids,y,anchor,p=m['fold'](fold);s023=.5*(anchor+p['all4'])
    def read(path):
        i,l,q=m['read'](path);np.testing.assert_array_equal(i,ids);np.testing.assert_array_equal(l,y);return q
    q=np.column_stack([anchor,*[p[k] for k in m['trees']],read(a/f'kaggle-t069/t069/{fold}-1-2400.npz'),read(a/f'kaggle-t070/t070/{fold}-gated-8.npz'),read(a/f'kaggle-t072/t072/{fold}-xgb-base.npz')]);w=np.r_[np.r_[.5*.83125,np.repeat(.125*.83125,4),.11875,.05]*.833,.167];b=.95*(.25*(q@w)+.75*expit(logit(np.clip(q,1e-8,1-1e-8))@w))+.05*read(a/f'kaggle-t081/t081/{fold}-quantile-8.npz');tree=read(a/f'kaggle-t083/t083/{fold}-paired.npz');base=.9*b+.1*tree;auc=roc_auc_score(y,base);assert abs(auc-{'earlier':.838853387,'primary':.842285180}[fold])<1e-8
    grid={};flags={'short':attrs['feature54'].to_numpy()[ids]<=60,'vector':attrs['present'].to_numpy()[ids]};weights=[0,.05,.10,.167,.25];
    for name,flag in flags.items():
        for lo in weights:
            for hi in weights:
                ww=np.where(flag,hi,lo);grid[f'{name}:{lo}:{hi}']=roc_auc_score(y,(1-ww)*b+ww*tree)
    if fold=='earlier':key=max(grid,key=grid.get);ctrl=max((k for k in grid if k.split(':')[1]==k.split(':')[2]),key=grid.get)
    result[fold]=dict(key=key,control=ctrl,auc=grid[key],gain=grid[key]-auc,gain_vs_control=grid[key]-grid[ctrl],gain_vs_S023=grid[key]-roc_auc_score(y,s023),grid=grid)
result['promote']=result['primary']['gain_vs_S023']>=.0003 and all(result[f]['gain']>0 for f in ['earlier','primary']);(t/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(key,[(f,result[f]['gain'],result[f]['gain_vs_S023']) for f in ['earlier','primary']],result['promote'])
