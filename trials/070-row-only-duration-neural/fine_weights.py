"""Post-hoc weight sensitivity. Freeze selection on earlier fold before evaluating primary."""
import json,runpy
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

t=Path(__file__).resolve().parent;art=t.parents[1]/'artifacts';m=runpy.run_path(str(t.parent/'065-row-only-blend-weights/run.py'));weights=[0,.005,.01,.025,.05,.075,.10,.125,.175,.20,.25,.375,.5,.625,.75,.875,1];report={}
for fold in ['earlier','primary']:
    ids,y,a,p=m['fold'](fold)
    def load(path):
        i,l,q=m['read'](path);np.testing.assert_array_equal(ids,i);np.testing.assert_array_equal(y,l);return q
    b=.875*(.5*a+.5*p['all4'])+.125*load(art/f'kaggle-t069/t069/{fold}-1-2400.npz');auc=roc_auc_score(y,b);grid={}
    for epoch in (2,5,8):
        nets={k:load(art/f'kaggle-t070/t070/{fold}-{k}-{epoch}.npz') for k in ['residual','gated']};nets['mean']=(nets['residual']+nets['gated'])/2
        for k,q in nets.items():
            for w in weights:grid[f'{k}/{epoch}/{w}']=float(roc_auc_score(y,(1-w)*b+w*q)-auc) if w else 0.
    if fold=='earlier':winner=max(grid,key=grid.get);original=max((k for k in grid if k.startswith('gated/5/')),key=grid.get)
    report[fold]=dict(base=auc,frozen=winner,gain=grid[winner],gated5_frozen=original,gated5_gain=grid[original],grid=grid)
(t/'fine-weights.json').write_text(json.dumps(report,indent=2)+'\n');print([(f,report[f]['frozen'],report[f]['gain'],report[f]['gated5_frozen'],report[f]['gated5_gain']) for f in report]);print([(w,*[report[f]['grid'][f'gated/5/{w}'] for f in report]) for w in [0,.01,.025,.05,.075,.125,.25]])
