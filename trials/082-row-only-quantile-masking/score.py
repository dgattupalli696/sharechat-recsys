from pathlib import Path
import json,sys,runpy
import numpy as np
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score
t=Path(__file__).resolve().parent;root=t.parents[1];art=root/'artifacts';m=runpy.run_path(str(t.parent/'065-row-only-blend-weights/run.py'));folder=Path(sys.argv[1]);result={}
for fold in ['earlier','primary']:
    ids,y,anchor,p=m['fold'](fold)
    def load(path):
        i,l,q=m['read'](path);np.testing.assert_array_equal(i,ids);np.testing.assert_array_equal(l,y);return q
    q=np.column_stack([anchor,*[p[k] for k in m['trees']],load(art/f'kaggle-t069/t069/{fold}-1-2400.npz'),load(art/f'kaggle-t070/t070/{fold}-gated-8.npz'),load(art/f'kaggle-t072/t072/{fold}-xgb-base.npz')]);w=np.r_[np.r_[.5*.83125,np.repeat(.125*.83125,4),.11875,.05]*.833,.167];base=.25*(q@w)+.75*expit(logit(np.clip(q,1e-8,1-1e-8))@w);auc=roc_auc_score(y,base);assert abs(auc-{'primary':.84223384,'earlier':.83873085}[fold])<1e-7;base=.95*base+.05*load(art/f'kaggle-t081/t081/{fold}-quantile-8.npz');auc=roc_auc_score(y,base);assert abs(auc-{'earlier':.83883482735,'primary':.84224752122}[fold])<1e-8;grid={}
    for ep in (2,5,8):
        z={k:load(folder/f'{fold}-{k}-{ep}.npz') for k in ['field','group']};z['mean']=(z['field']+z['group'])/2;z['control']=load(folder/f'{fold}-control-{ep}.npz')
        for k,v in z.items():
            for f in [0,.025,.05,.10,.167,.25,.375]:grid[f'{k}/{ep}/{f}']=float(roc_auc_score(y,(1-f)*base+f*v)) if f else auc
    if fold=='earlier':key=max(grid,key=grid.get);ctrl=max((k for k in grid if k.startswith('control/')),key=grid.get)
    result[fold]=dict(selected=key,auc=grid[key],delta=grid[key]-auc,gain_vs_control=grid[key]-grid[ctrl],gain_vs_S023=grid[key]-roc_auc_score(y,.5*(anchor+p['all4'])),grid=grid)
result['promote']=result['primary']['gain_vs_S023']>=.0003 and all(result[f]['delta']>0 for f in ['earlier','primary']);(t/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(key,result['primary']['delta'],result['promote'])
