import json
from pathlib import Path
import sys
import numpy as np
from sklearn.metrics import roc_auc_score

t=Path(__file__).resolve().parent;root=t.parents[1];folder=Path(sys.argv[1]);results={}
def read(path):
    with np.load(path) as z:
        ix=z['row_ids'].argsort();return z['row_ids'][ix],z['y_true'].ravel()[ix],z['probabilities'].ravel()[ix]
for fold in ('primary','earlier'):
    ref=(root/'artifacts/local-t016/validation_predictions.npz' if fold=='primary' else root/'artifacts/local-t018-confirmation-analysis/earlier/frozen-75-25.npz');ids,y,anchor=read(ref);scores={}
    for depth,rich in [(12,0),(12,1),(6,0),(6,1)]:
        k=f'd{depth}-r{rich}';i,labels,p=read(folder/f'{fold}-{k}.npz');np.testing.assert_array_equal(i,ids);np.testing.assert_array_equal(labels,y);auc=roc_auc_score(y,.75*anchor+.25*p);scores[k]={'auc':roc_auc_score(y,p),'blend':auc,'delta_vs_matched_control':auc-scores.get('d12-r0',{'blend':auc})['blend']}
    if fold=='primary':
        # Use actual S022 GPU holdout predictions as well as replay, so backend noise can't masquerade as a gain.
        i,labels,p=read(root/'artifacts/kaggle-t062/t062/depth12-holdout.npz');np.testing.assert_array_equal(i,ids);np.testing.assert_array_equal(labels,y);s022=roc_auc_score(y,.75*anchor+.25*p)
        for v in scores.values():v['delta_vs_S022']=v['blend']-s022
    results[fold]=scores
eligible=[k for k in results['primary'] if k!='d12-r0' and results['primary'][k]['delta_vs_S022']>=.0003 and results['primary'][k]['delta_vs_matched_control']>0 and results['earlier'][k]['delta_vs_matched_control']>0];results['refit_candidates']=eligible;(t/'results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2))
