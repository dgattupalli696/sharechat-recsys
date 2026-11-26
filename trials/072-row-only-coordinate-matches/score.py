import json,sys,runpy
from pathlib import Path
from sklearn.metrics import roc_auc_score
import numpy as np
t=Path(__file__).resolve().parent;root=t.parents[1];a=root/'artifacts';m=runpy.run_path(str(t.parent/'065-row-only-blend-weights/run.py'));w=json.loads((t.parent/'071-row-only-categorical-interactions/carry.json').read_text());folder=Path(sys.argv[1]);result={}
for fold in ['earlier','primary']:
    ids,y,anchor,p=m['fold'](fold);s023=.5*(anchor+p['all4']);
    def read(path):
        i,l,q=m['read'](path);np.testing.assert_array_equal(i,ids);np.testing.assert_array_equal(l,y);return q
    b=w['s023']*s023+w['t069_duration_depth6_round2400']*read(a/f'kaggle-t069/t069/{fold}-1-2400.npz')+w['t070_gated_epoch8']*read(a/f'kaggle-t070/t070/{fold}-gated-8.npz');auc=roc_auc_score(y,b);assert abs(auc-{'primary':.84211014,'earlier':.83862609}[fold])<1e-7
    arms={k:read(folder/f'{fold}-{k}.npz') for k in ['xgb-base','xgb-products','cat-base','cat-products']};arms['products-mean']=(arms['xgb-products']+arms['cat-products'])/2;arms['base-mean']=(arms['xgb-base']+arms['cat-base'])/2;grid={f'{k}/{v}':(float(roc_auc_score(y,(1-v)*b+v*q)) if v else auc) for k,q in arms.items() for v in (0,.025,.05,.1,.167,.25,.375,.5)}
    if fold=='earlier':choices={k:max((j for j in grid if j.startswith(k+'/')),key=grid.get) for k in arms};winner=max(grid,key=grid.get)
    result[fold]=dict(selected=winner,auc=grid[winner],gain=grid[winner]-auc,gain_vs_S023=grid[winner]-roc_auc_score(y,s023),arm_deltas={k:grid[j]-auc for k,j in choices.items()},grid=grid)
result['promote']=result['primary']['gain_vs_S023']>=.0003 and all(result[f]['gain']>0 for f in ['earlier','primary']);(t/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(winner,result['primary']['gain'],result['primary']['gain_vs_S023'],result['promote'])
