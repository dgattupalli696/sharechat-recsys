"""Historical, offline-only S017 comparison. No test set or submission file."""
import json,runpy
from pathlib import Path
import numpy as np
import polars as pl
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score as auc

t=Path(__file__).resolve().parent;root=t.parents[1];a=root/'artifacts';m=runpy.run_path(str(t.parent/'065-row-only-blend-weights/run.py'));ids,y,anchor,p=m['fold']('primary')
def read(tr,k):
    i,l,v=m['read'](a/f'kaggle-t{tr}/t{tr}/primary-{k}.npz');np.testing.assert_array_equal(ids,i);np.testing.assert_array_equal(y,l);return v
duration=read('069','1-2400');neural=read('081','quantile-8');paired=read('083','paired');q=np.column_stack([anchor,*[p[k] for k in m['trees']],duration,read('070','gated-8'),read('072','xgb-base')]);s024=runpy.run_path(str(t.parent/'087-full-row-only-landmark-blend/assemble.py'))['blend'](q,neural,paired,read('086','learned'),read('086','random'));z=np.load(a/'local-t052/teacher-blend.npz');i=z['row_ids'].argsort();np.testing.assert_array_equal(ids,z['row_ids'][i]);old=z['probabilities'][i];assert abs(auc(y,old)-.89105958)<1e-7 and abs(auc(y,s024)-.842296286)<1e-7
meta=pl.read_parquet(root/'data/train.parquet',columns=['feature1','request_id']);ts=meta['feature1'].cast(pl.Int64).to_numpy()[ids];first=ts<np.median(ts);ws=np.array([0,.05,.1,.2,.3,.4,.5,.6,.75,1,1.25,1.5,2,2.5,3,4,5,7.5,10,12.5,16.7,20,25,33,50,75,100])/100
pred={'S024':s024,'quantile':neural,'paired':paired,'duration':duration};report={}; plans={}
for name,mask in [('selection',first),('evaluation',~first)]:
    y0=y[mask];r=rankdata(old[mask]);rs={k:rankdata(v[mask]) for k,v in pred.items()};rs['quantile_duration_mean']=(rs['quantile']+rs['duration'])/2;ref=auc(y0,r);grid={k:{f'{100*w:g}':float(auc(y0,(1-w)*r+w*v)-ref) for w in ws} for k,v in rs.items()};report[name]=dict(n=len(y0),baseline=ref,delta=grid)
    if name=='selection':
        plans={k:max(g,key=g.get) for k,g in grid.items()};winner=max(plans,key=lambda k:grid[k][plans[k]])
    else:
        from importlib.util import spec_from_file_location,module_from_spec
        spec=spec_from_file_location('prev',t.parent/'061-row-only-temporal-confirmation/run.py');prev=module_from_spec(spec);spec.loader.exec_module(prev);w=float(plans[winner])/100;groups=np.unique(meta['request_id'].to_numpy()[ids][mask],return_inverse=True)[1];report['winner_ci']=prev.ci(y0,r,(1-w)*r+w*rs[winner],groups)
report['frozen_percent']=plans;report['winner']=winner;(t/'results.json').write_text(json.dumps(report,indent=2)+'\n');print('baselines',report['selection']['baseline'],report['evaluation']['baseline']);print([(k,v,report['selection']['delta'][k][v],report['evaluation']['delta'][k][v]) for k,v in plans.items()],winner,report['winner_ci']);print('quantile sensitivity',[(w,report['evaluation']['delta']['quantile'][w]) for w in ['0.1','0.5','1','2','5','10','25','50']])
