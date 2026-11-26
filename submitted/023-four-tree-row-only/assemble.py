from pathlib import Path
import sys,json,hashlib
import polars as pl
import numpy as np
t=Path(__file__).resolve().parent;root=t.parents[1];folder=Path(sys.argv[1]);ids=pl.read_csv(root/'data/sample_submission.csv')['id'];assert len(ids)==ids.n_unique()==1037090
def read(path):
    f=pl.read_csv(root/path);np.testing.assert_array_equal(f['id'].to_numpy(),ids.to_numpy());return f['watched'].to_numpy()
a=read('artifacts/kaggle-t021/refit/submission-full-ensemble.csv');p=[read('artifacts/kaggle-t062/t062/submission-full-depth12.csv')]
for k in ['d12-r1','d6-r0','d6-r1']:
    d,r=k[1:].split('-r');m=json.loads((folder/f'full-{d}-{r}.json').read_text());assert m['fit']==1537638 and m['test']==len(ids);p.append(np.load(folder/f'full-{k}.npy'))
for v in [a,*p]:assert v.shape==(len(ids),) and np.isfinite(v).all() and ((v>=0)&(v<=1)).all()
blend=.5*a+.125*sum(p);dest=root/'artifacts/local-t066/submission-rowonly-half-four-trees.csv';dest.parent.mkdir(exist_ok=True,parents=True);pl.DataFrame({'id':ids,'watched':blend}).write_csv(dest);np.testing.assert_allclose(read(dest),blend,rtol=0,atol=1e-15);report={'path':str(dest.relative_to(root)),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'rows':len(ids)};(t/'candidate.json').write_text(json.dumps(report,indent=2)+'\n');print(report)
