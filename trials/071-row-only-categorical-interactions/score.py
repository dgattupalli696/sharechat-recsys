import json,sys,runpy
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
t=Path(__file__).resolve().parent;root=t.parents[1];m=runpy.run_path(str(t.parent/'065-row-only-blend-weights/run.py'));folder=Path(sys.argv[1]);w0=json.loads((t/'carry.json').read_text());report={};arms=['control','additive','cross'];weights=[0,.01,.025,.05,.075,.125,.25,.375]
for fold in ['earlier','primary']:
    ids,y,a,p=m['fold'](fold);s023=.5*a+.5*p['all4']
    def read(path):
        i,l,q=m['read'](path);np.testing.assert_array_equal(i,ids);np.testing.assert_array_equal(l,y);return q
    b=w0['s023']*s023+w0['t069_duration_depth6_round2400']*read(root/f'artifacts/kaggle-t069/t069/{fold}-1-2400.npz')+w0['t070_gated_epoch8']*read(root/f'artifacts/kaggle-t070/t070/{fold}-gated-8.npz');ba=roc_auc_score(y,b);assert abs(ba-{'primary':.84211014,'earlier':.83862609}[fold])<1e-7;grid={}
    for arm in arms:
        for epoch in [5,8]:
            q=read(folder/f'{fold}-{arm}-{epoch}.npz')
            for w in weights:grid[f'{arm}/{epoch}/{w}']=float(roc_auc_score(y,(1-w)*b+w*q)) if w else ba
    if fold=='earlier':
        selections={arm:max((k for k in grid if k.startswith(arm+'/')),key=grid.get) for arm in arms};winner=max(grid,key=grid.get)
    report[fold]=dict(base=ba,selected=winner,auc=grid[winner],gain=grid[winner]-ba,gain_vs_S023=grid[winner]-roc_auc_score(y,s023),gain_vs_tuned_control=grid[winner]-grid[selections['control']],arms={a:{'choice':k,'gain':grid[k]-ba} for a,k in selections.items()},grid=grid)
report['promote']=report['primary']['gain_vs_S023']>=.0003 and all(report[f]['gain']>0 for f in ['earlier','primary']);(t/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(winner,[(f,report[f]['gain'],report[f]['gain_vs_S023']) for f in ['earlier','primary']],report['promote'])
