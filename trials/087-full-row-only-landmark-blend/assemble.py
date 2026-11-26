from pathlib import Path
import json,hashlib,sys
import polars as pl
import numpy as np
from scipy.special import expit,logit

t=Path(__file__).resolve().parent;root=t.parents[1];a=root/'artifacts'
def blend(q,quantile,paired,learned,random):
    w=np.r_[np.r_[.5*.83125,np.repeat(.125*.83125,4),.11875,.05]*.833,.167];b=.25*(q@w)+.75*expit(logit(np.clip(q,1e-8,1-1e-8))@w);return .95*(.9*(.95*b+.05*quantile)+.1*paired)+.025*(learned+random)

def assemble(folder):
    ids=pl.read_csv(root/'data/sample_submission.csv')['id'];np.testing.assert_array_equal(ids.to_numpy(),np.load(folder/'ids.npy'));assert len(ids)==ids.n_unique()==1037090
    def read(path):
        f=pl.read_csv(path);np.testing.assert_array_equal(ids.to_numpy(),f['id'].to_numpy());return f['watched'].to_numpy()
    def check(v):assert v.shape==(len(ids),) and np.isfinite(v).all() and ((v>=0)&(v<=1)).all();return v
    q=[read(a/'kaggle-t021/refit/submission-full-ensemble.csv'),read(a/'kaggle-t062/t062/submission-full-depth12.csv'),*[np.load(a/f'kaggle-t066/t066/full-{k}.npy') for k in ['d12-r1','d6-r0','d6-r1']]];np.testing.assert_allclose(.5*q[0]+.125*sum(q[1:]),read(a/'local-t066/submission-rowonly-half-four-trees.csv'),rtol=0,atol=1e-12)
    def new(k):
        state=json.loads((folder/f'{k}.json').read_text());assert state['fit']==1537638 and state['test']==len(ids);return check(np.load(folder/f'{k}.npy'))
    q=np.column_stack([*map(check,q),*[new(k) for k in ['duration','gated','compact']]]);p=check(blend(q,*[new(k) for k in ['quantile','paired','learned','random']]));dest=a/'local-t087/submission-full-rowonly-landmarks.csv';dest.parent.mkdir(exist_ok=True,parents=True);pl.DataFrame({'id':ids,'watched':p}).write_csv(dest);np.testing.assert_allclose(read(dest),p,atol=1e-15,rtol=0);rep={'file':str(dest.relative_to(root)),'rows':len(p),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()};(t/'candidate.json').write_text(json.dumps(rep,indent=2)+'\n');print(rep)
if __name__=='__main__':assemble(Path(sys.argv[1]))
