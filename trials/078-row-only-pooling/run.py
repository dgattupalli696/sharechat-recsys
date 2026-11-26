"""Earlier-selected pooling rule, with incumbent components and weights fixed."""
import json,runpy
from pathlib import Path
import numpy as np
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score
t=Path(__file__).resolve().parent;root=t.parents[1];art=root/'artifacts';m=runpy.run_path(str(t.parent/'065-row-only-blend-weights/run.py'));report={}
for fold in ['earlier','primary']:
    ids,y,anchor,p=m['fold'](fold)
    def load(path):
        i,l,q=m['read'](path);np.testing.assert_array_equal(ids,i);np.testing.assert_array_equal(l,y);return q
    q=np.column_stack([anchor,*[p[k] for k in m['trees']],load(art/f'kaggle-t069/t069/{fold}-1-2400.npz'),load(art/f'kaggle-t070/t070/{fold}-gated-8.npz'),load(art/f'kaggle-t072/t072/{fold}-xgb-base.npz')]);w=np.r_[.5*.83125,np.repeat(.125*.83125,4),.11875,.05]*.833;w=np.r_[w,.167];np.testing.assert_allclose(w.sum(),1);s023=.5*(anchor+p['all4']);base=q@w;auc=roc_auc_score(y,base);assert abs(auc-{'earlier':.838714684,'primary':.842213528}[fold])<1e-8
    def scores(part):
        z=np.clip(part,1e-8,1-1e-8);ar=part@w;geo=expit(logit(z)@w);return {**{f'logit/{a}':(1-a)*ar+a*geo for a in [.25,.5,.75,1]},**{f'power/{k}':((z**k)@w)**(1/k) for k in [.25,.5,1,1.5,2]},'baseline':ar}
    grid={k:roc_auc_score(y,s) for k,s in scores(q).items()};
    if fold=='earlier':best=max(grid,key=grid.get)
    np.testing.assert_allclose(scores(q[:12])[best],np.array([scores(row[None,:])[best].item() for row in q[:12]]),atol=1e-12);report[fold]=dict(choice=best,auc=grid[best],gain=grid[best]-auc,gain_vs_S023=grid[best]-roc_auc_score(y,s023),grid=grid)
report['promote']=report['primary']['gain_vs_S023']>=.0003 and all(report[f]['gain']>0 for f in ['earlier','primary']);(t/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(best,[(f,report[f]['gain'],report[f]['gain_vs_S023']) for f in ['earlier','primary']],report['promote'])
