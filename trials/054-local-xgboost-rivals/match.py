"""D: learn observed vs matched swapped request-item pairs, without watch labels."""
import argparse
import json
import subprocess
import sys

import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from run import CUTOFF, OUT, ROOT, TRIAL, compare, load, unit_vectors

PAIRS = [('feature32','feature43'), ('feature30','feature46'), ('feature31','feature44')]


def swaps(pool, item, rq, hour, duration, freq, exposed, stride, rng):
    # Shuffle independently within exact hour, duration bin, and log2 popularity bin.
    # Reject any donor from the recipient request or already on its observed slate.
    strata=np.stack([hour[pool],duration[pool],np.floor(np.log2(np.maximum(1,freq[item[pool]]))).astype(int)],axis=1)
    order=np.lexsort((rng.random(len(pool)),*strata.T)); src=pool[order]; s=strata[order]
    starts=np.r_[0,np.flatnonzero(np.any(s[1:]!=s[:-1],axis=1))+1,len(src)]
    dest=np.empty_like(src)
    for a,b in zip(starts[:-1],starts[1:]): dest[a:b]=np.roll(src[a:b],rng.integers(1,b-a) if b-a>1 else 0)
    ok=(rq[src]!=rq[dest]) & ~np.isin(rq[src]*stride+item[dest],exposed)
    return src[ok],dest[ok]


def prepare(smoke=False):
    lf=pl.scan_parquet(ROOT/'data/train.parquet').with_row_index('row')
    if smoke: lf=lf.filter(pl.col('request_id').hash(42)%32==0)
    f=lf.select('row','request_id','feature18','feature54','feature1',*[c for p in PAIRS for c in p]).collect()
    valid=(f['feature1']>=CUTOFF).to_numpy(); fold=(f['request_id'].hash(17).to_numpy()%3).astype(int)
    if not smoke: np.testing.assert_array_equal(f['row'].to_numpy()[valid],load('row_ids'))
    rq=np.unique(f['request_id'].to_numpy(),return_inverse=True)[1]; item=np.unique(f['feature18'].to_numpy(),return_inverse=True)[1]; stride=int(item.max()+1)
    exposed=np.unique(rq*stride+item); duration=np.digitize(f['feature54'].to_numpy(),[50,60,75,90,120,180,240]); hour=f['feature1'].dt.epoch('s').to_numpy()//3600
    users=[unit_vectors(f[u]) for u,_ in PAIRS]; videos=[unit_vectors(f[v]) for _,v in PAIRS]; scores=np.full(len(f),np.nan,np.float32); audits=[]
    for k in range(3):
        rng=np.random.default_rng(42+k); tr=np.flatnonzero(~valid & (fold!=k)); score=np.flatnonzero(fold==k)
        uniq,first=np.unique(item[tr],return_index=True); lookup=np.full(stride,-1); lookup[uniq]=tr[first]; freq=np.bincount(item[tr],minlength=stride)
        def features(src,dst):
            # Use one canonical donor representation per known item, in real and swapped pairs alike.
            j=lookup[item[dst]]; j=np.where(j>=0,j,dst)
            return np.column_stack([32*u[src]*v[j] for u,v in zip(users,videos)])
        src,dst=swaps(tr,item,rq,hour,duration,freq,exposed,stride,rng); coverage=len(src)/len(tr)
        sel=rng.choice(len(src),min(160000,len(src)),replace=False); src,dst=src[sel],dst[sel]
        x=np.concatenate([features(src,src),features(src,dst)]); y=np.repeat([1,0],len(src)); model=LogisticRegression(C=.1,max_iter=150,random_state=42).fit(x,y); del x
        for start in range(0,len(score),16384):
            p=score[start:start+16384]; scores[p]=model.decision_function(features(p,p))
        # Diagnose discrimination on unseen pre-cutoff requests, not on training pairs.
        a,b=swaps(score[~valid[score]],item,rq,hour,duration,freq,exposed,stride,rng)
        if len(a)>30000:
            take=rng.choice(len(a),30000,replace=False); a,b=a[take],b[take]
        diagnostic=roc_auc_score(np.repeat([1,0],len(a)),np.r_[scores[a],model.decision_function(features(a,b))]); audit=dict(fold=k,pairs=len(src),matched_fraction=coverage,unseen_request_pair_auc=diagnostic); audits.append(audit);print(audit,flush=True)
    assert np.isfinite(scores).all()
    ranks=pl.DataFrame({'rq':rq,'s':scores}).select(pl.col('s').rank('average').over('rq')/pl.len().over('rq')).to_series().to_numpy(); extra=np.column_stack([scores,ranks]).astype('f4')
    if not smoke:
        for mask,split in [(valid,'valid'),(~valid,'fit')]: np.save(OUT/f'match_{split}.npy',extra[mask])
        (TRIAL/'match-diagnostics.json').write_text(json.dumps(audits,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--prepare',action='store_true');args=parser.parse_args()
    if args.smoke or args.prepare: prepare(args.smoke)
    else:
        for script,arg,done in [('match.py','--prepare',TRIAL/'match-diagnostics.json'),('run.py','D',TRIAL/'D.json')]:
            if not done.exists(): subprocess.run([sys.executable,'-u',str(TRIAL/script),arg],check=True,cwd=ROOT)
        compare(match=True)
