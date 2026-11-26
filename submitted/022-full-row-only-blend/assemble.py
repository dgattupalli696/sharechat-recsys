import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

TRIAL=Path(__file__).resolve().parent;ROOT=TRIAL.parents[1]; OLD=ROOT/'artifacts/kaggle-t021/refit'


def main(folder=None):
    v=json.loads((TRIAL.parent/'021-full-data-finalist-refit/independent-verification.json').read_text());sample=pl.read_csv(ROOT/'data/sample_submission.csv');out=ROOT/'artifacts/local-t062';out.mkdir(exist_ok=True,parents=True)
    def read(path):
        f=pl.read_csv(path);assert f.columns==['id','watched'] and len(f)==1037090 and f['id'].n_unique()==len(f);np.testing.assert_array_equal(sample['id'].to_numpy(),f['id'].to_numpy());p=f['watched'].to_numpy();assert np.isfinite(p).all() and ((0<=p)&(p<=1)).all();return p
    report={};assert v['full_training_rows']==1537638
    for family in ('ensemble','tabm'):
        rec=v['submissions'][family];path=OLD/rec['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==rec['sha256'];read(path);report[family]=rec
    if folder:
        folder=Path(folder);p=read(folder/'submission-full-depth12.csv');assert json.loads((folder/'depth12-full.json').read_text())['full_train_rows']==1537638
        z=np.load(ROOT/'artifacts/local-t016/validation_predictions.npz');ix=np.argsort(z['row_ids']);ids=z['row_ids'][ix];y=z['y_true'].ravel()[ix];base=z['probabilities'].ravel()[ix];baseline=roc_auc_score(y,base);assert abs(baseline-.8408057533)<1e-8
        for name in ['depth12','leaves511','catboost-ids']:
            with np.load(folder/f'{name}-holdout.npz') as w:
                sort=w['row_ids'].argsort();np.testing.assert_array_equal(ids,w['row_ids'][sort]);np.testing.assert_array_equal(y,w['y_true'][sort]);q=w['probabilities'][sort];auc=roc_auc_score(y,.75*base+.25*q);report[name]={'auc':roc_auc_score(y,q),'fixed_blend_auc':auc,'delta':auc-baseline}
        blend=.75*read(OLD/v['submissions']['ensemble']['file'])+.25*p;path=out/'submission-full-rowonly-75-25.csv';sample.with_columns(pl.Series('watched',blend)).write_csv(path);np.testing.assert_allclose(read(path),blend,atol=1e-15,rtol=0);report['candidate']={'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':len(p)}
    (TRIAL/('results.json' if folder else 'baseline-check.json')).write_text(json.dumps(report,indent=2)+'\n');print(report)


if __name__=='__main__':main(*sys.argv[1:])
