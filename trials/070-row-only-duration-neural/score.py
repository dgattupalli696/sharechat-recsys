import json,sys,runpy
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
t=Path(__file__).resolve().parent;root=t.parents[1];m=runpy.run_path(str(t.parent/'065-row-only-blend-weights/run.py'));folder=Path(sys.argv[1]);result={}
for fold in ['earlier','primary']:
    ids,y,a,p=m['fold'](fold);base=.5*a+.5*p['all4']
    def read(path):
        i,l,q=m['read'](path);np.testing.assert_array_equal(i,ids);np.testing.assert_array_equal(l,y);return q
    duration=read(root/f'artifacts/kaggle-t069/t069/{fold}-1-2400.npz');carry=.875*base+.125*duration;base_auc=roc_auc_score(y,base);carry_auc=roc_auc_score(y,carry);grid={}
    for epoch in (2,5,8):
        models={k:read(folder/f'{fold}-{k}-{epoch}.npz') for k in ['residual','gated']};models['mean']=(models['residual']+models['gated'])/2
        for k,q in models.items():
            for w in (.125,.25,.375,.5):grid[f'{k}/{epoch}/{w}']=float(roc_auc_score(y,(1-w)*carry+w*q))
    if fold=='earlier':winner=max(grid,key=grid.get)
    auc=grid[winner];result[fold]=dict(s023_auc=base_auc,carry_auc=carry_auc,selected=winner,blend_auc=auc,delta_vs_s023=auc-base_auc,delta_vs_carry=auc-carry_auc,grid=grid)
result['promote']=result['primary']['delta_vs_s023']>=.0003 and all(result[f]['delta_vs_carry']>0 for f in ('primary','earlier'));(t/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(winner,[(f,result[f]['delta_vs_s023'],result[f]['delta_vs_carry']) for f in ('earlier','primary')],result['promote'])
