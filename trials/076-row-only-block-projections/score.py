from pathlib import Path
import sys,json,runpy
import numpy as np
from sklearn.metrics import roc_auc_score
t=Path(__file__).resolve().parent;root=t.parents[1];a=root/'artifacts';folder=Path(sys.argv[1]);m=runpy.run_path(str(t.parent/'065-row-only-blend-weights/run.py'));result={}
for fold in ['earlier','primary']:
    ids,y,anchor,p=m['fold'](fold);s023=.5*(anchor+p['all4'])
    def read(path):
        i,l,q=m['read'](path);np.testing.assert_array_equal(i,ids);np.testing.assert_array_equal(l,y);return q
    carry=.83125*s023+.11875*read(a/f'kaggle-t069/t069/{fold}-1-2400.npz')+.05*read(a/f'kaggle-t070/t070/{fold}-gated-8.npz');q0=read(a/f'kaggle-t072/t072/{fold}-xgb-base.npz');base=.833*carry+.167*q0;auc=roc_auc_score(y,base);assert abs(auc-{'primary':.842213528,'earlier':.838714684}[fold])<1e-8
    arms={'depth-127':q0,**{k:read(folder/f'{fold}-{k}.npz') for k in ['pca','random']}};arms['mean']=np.mean(list(arms.values()),axis=0);grid={f'{k}/{w}':float(roc_auc_score(y,(1-w)*carry+w*q)) for k,q in arms.items() for w in [.10,.167,.25,.375,.5]}
    if fold=='earlier':key=max(grid,key=grid.get);control=max((k for k in grid if k.startswith('depth-127/')),key=grid.get)
    result[fold]=dict(selected=key,auc=grid[key],gain_vs_T072=grid[key]-auc,gain_vs_weight_control=grid[key]-grid[control],gain_vs_S023=grid[key]-roc_auc_score(y,s023),grid=grid)
result['promote']=result['primary']['gain_vs_S023']>=.0003 and all(result[f]['gain_vs_T072']>0 for f in ['earlier','primary']);(t/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(key,[(f,result[f]['gain_vs_S023']) for f in ['earlier','primary']],result['promote'])
