import json,runpy,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
t=Path(__file__).resolve().parent;m=runpy.run_path(str(t.parent/'065-row-only-blend-weights/run.py'));folder=Path(sys.argv[1]);results={};weights=[.125,.25,.375,.5];plans={}
for fold in ['earlier','primary']:
    ids,y,a,p=m['fold'](fold);b=.5*a+.5*p['all4'];baseline=roc_auc_score(y,b);assert abs(baseline-{'primary':.84198952,'earlier':.83851422}[fold])<1e-7
    table={}
    for e in [0,1]:
        for r in [900,1600,2400]:
            i,l,q=m['read'](folder/f'{fold}-{e}-{r}.npz');np.testing.assert_array_equal(i,ids);np.testing.assert_array_equal(l,y)
            if (e,r)==(0,900):assert abs(roc_auc_score(y,q)-roc_auc_score(y,p['d6-r1']))<.0001,'control replay mismatch'
            for w in weights:
                key=f'{e}/{r}/{w}';s=(1-w)*b+w*q;table[key]=float(roc_auc_score(y,s)-baseline)
    if fold=='earlier':
        for label,subset in [('weight_only',[k for k in table if k.startswith('0/900/')]),('longer_only',[k for k in table if k.startswith('0/') and not k.startswith('0/900/')]),('duration_900',[k for k in table if k.startswith('1/900/')]),('duration_long',[k for k in table if k.startswith('1/') and not k.startswith('1/900/')])]:plans[label]=max(subset,key=table.get)
        plans['overall']=max(table,key=table.get)
    results[fold]={'baseline':baseline,'frozen_deltas':{name:table[k] for name,k in plans.items()},'grid':table}
results['frozen_plans']=plans;results['promote']=results['primary']['frozen_deltas']['overall']>=.0003 and results['earlier']['frozen_deltas']['overall']>0;(t/'results.json').write_text(json.dumps(results,indent=2)+'\n');print(plans,results['primary']['frozen_deltas'],results['promote'])
