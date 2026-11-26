import json,sys,runpy
from pathlib import Path
import numpy as np,polars as pl
from sklearn.metrics import roc_auc_score
t=Path(__file__).resolve().parent;root=t.parents[1];a=root/'artifacts';folder=Path(sys.argv[1]);m=runpy.run_path(str(t.parent/'065-row-only-blend-weights/run.py'));d=pl.read_parquet(root/'data/train.parquet',columns=['feature54'])['feature54'].to_numpy();result={}
for fold in ['earlier','primary']:
    ids,y,anchor,p=m['fold'](fold);s023=.5*(anchor+p['all4']);short=np.isfinite(d[ids])&(d[ids]<=60)
    def read(path):
        i,l,q=m['read'](path);np.testing.assert_array_equal(i,ids);np.testing.assert_array_equal(l,y);return q
    carry=.83125*s023+.11875*read(a/f'kaggle-t069/t069/{fold}-1-2400.npz')+.05*read(a/f'kaggle-t070/t070/{fold}-gated-8.npz');base=.833*carry+.167*read(a/f'kaggle-t072/t072/{fold}-xgb-base.npz');auc=roc_auc_score(y,base);assert abs(auc-{'earlier':.838714684,'primary':.842213528}[fold])<1e-8
    grid={}
    for kind in ['duration','random']:
        small,large=[read(folder/f'{fold}-{kind}-{j}.npz') for j in ['short','long']];q=np.where(short,small,large)
        for w in [0,.025,.05,.10,.167,.25,.375]:grid[f'{kind}/{w}']=float(roc_auc_score(y,(1-w)*base+w*q))
    if fold=='earlier':winner=max((k for k in grid if k.startswith('duration/')),key=grid.get);ctrl=max((k for k in grid if k.startswith('random/')),key=grid.get)
    result[fold]=dict(winner=winner,control=ctrl,auc=grid[winner],gain_vs_T072=grid[winner]-auc,gain_vs_random=grid[winner]-grid[ctrl],gain_vs_S023=grid[winner]-roc_auc_score(y,s023),grid=grid)
result['promote']=result['primary']['gain_vs_S023']>=.0003 and all(result[f]['gain_vs_T072']>0 and result[f]['gain_vs_random']>0 for f in ['earlier','primary']);(t/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(winner,ctrl,result['primary']['gain_vs_S023'],result['promote'])
