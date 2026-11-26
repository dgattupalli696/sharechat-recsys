from pathlib import Path
import json,sys,runpy
import numpy as np
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score
t=Path(__file__).resolve().parent;root=t.parents[1];a=root/'artifacts';m=runpy.run_path(str(t.parent/'065-row-only-blend-weights/run.py'));folder=Path(sys.argv[1]);result={}
for fold in ['earlier','primary']:
    ids,y,anchor,p=m['fold'](fold);s023=.5*(anchor+p['all4'])
    def read(path):
        i,l,q=m['read'](path);np.testing.assert_array_equal(i,ids);np.testing.assert_array_equal(l,y);return q
    q=np.column_stack([anchor,*[p[k] for k in m['trees']],read(a/f'kaggle-t069/t069/{fold}-1-2400.npz'),read(a/f'kaggle-t070/t070/{fold}-gated-8.npz'),read(a/f'kaggle-t072/t072/{fold}-xgb-base.npz')]);w=np.r_[np.r_[.5*.83125,np.repeat(.125*.83125,4),.11875,.05]*.833,.167];b=.95*(.25*(q@w)+.75*expit(logit(np.clip(q,1e-8,1-1e-8))@w))+.05*read(a/f'kaggle-t081/t081/{fold}-quantile-8.npz');auc=roc_auc_score(y,b);assert abs(auc-{'primary':.84224752,'earlier':.83883483}[fold])<1e-8
    original=read(a/f'kaggle-t083/t083/{fold}-paired.npz');ref=.9*b+.1*original;t083=roc_auc_score(y,ref);assert abs(t083-{'primary':.8422851797,'earlier':.8388533869}[fold])<1e-8
    arms={k:read(folder/f'{fold}-{k}.npz') for k in ['4','24']};arms['12']=original;arms['mean']=(original+arms['4']+arms['24'])/3;grid={f'{k}/{f}':float(roc_auc_score(y,(1-f)*b+f*z)) for k,z in arms.items() for f in [.05,.10,.167,.25]}
    if fold=='earlier':key=max(grid,key=grid.get);choices={k:max((j for j in grid if j.startswith(k+'/')),key=grid.get) for k in arms}
    result[fold]=dict(selected=key,auc=grid[key],gain=grid[key]-t083,gain_vs_S023=grid[key]-roc_auc_score(y,s023),arms={k:grid[j]-t083 for k,j in choices.items()},grid=grid)
result['promote']=result['primary']['gain_vs_S023']>=.0003 and all(result[f]['gain']>0 for f in ['earlier','primary']);(t/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(key,result['primary']['gain'],result['promote'])
