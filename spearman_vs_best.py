import glob,json,hashlib,re,numpy as np,polars as pl
from pathlib import Path
from scipy.stats import rankdata
H=json.load(open('artifacts/local-blend-search/results.json'))['models']
sub={}
for r in glob.glob('submitted/*/submission-receipt.json'):
    d=json.load(open(r)); s='S'+Path(r).parent.name[:3]
    sub[d.get('sha256') or d.get('submission_sha256')]=(s,d.get('public'))
pub={l.split('|')[1].strip():l.split('|')[5].strip().strip('*') for l in open('README.md') if l.startswith('| S0')}
def hold(p):
    m=re.search(r'kaggle-t039/t039/submission-(\w+?)-s42',p)
    if m: return H.get(f't036/{m[1]}'),'holdout (T036 cfg)'
    m=re.search(r'kaggle-(t040|t048)/\1/([^/]+)/',p)
    if m: return H.get(f'{m[1]}/{m[2]}'),'holdout'
    if 'kaggle-t027/' in p: return H['t027/ensemble-32'],'holdout'
    if 'kaggle-t028/' in p: return H['t028'],'holdout'
    m=re.search(r'kaggle-(s00\d-s00\d)/submission-(tabm|ensemble)',p)
    if m and m[2]=='tabm': return H[f'kaggle-{m[1]}/tabm32'],'holdout'
    oof={'t046/submission-depth12_node':0.88902,'t046/submission-leaves511_eta02':0.88889,'t046/submission-catboost_ids':0.88742,'t047/refit/submission-full-tabm-seed-mean':0.88893}
    for k,v in oof.items():
        if k in p: return v,'OOF'
    return None,'full refit/blend'
ref=pl.read_csv('artifacts/candidates/submission-s011-fusioncos.csv').sort('id')
ids=ref['id']; R=rankdata(ref['watched'].to_numpy())
rows=[]
for p in sorted(glob.glob('artifacts/**/*.csv',recursive=True)):
    if 'profile' in p or Path(p).stat().st_size<1e6: continue
    sha=hashlib.sha256(open(p,'rb').read()).hexdigest()
    d=pl.read_csv(p).sort('id')
    if d.height!=ref.height or not (d['id']==ids).all(): rows.append((p,None,None,'',None,'id mismatch'));continue
    rho=np.corrcoef(R,rankdata(d['watched'].to_numpy()))[0,1]
    s,pu=sub.get(sha,('',None)); pu=pu or (pub.get(s) if s else None)
    h,kind=hold(p)
    rows.append((p.replace('artifacts/',''),h,kind,s,pu,round(rho,5)))
rows.sort(key=lambda r:-(r[5] if isinstance(r[5],float) else 0))
print('| file | holdout | kind | sub | public | spearman vs S017 |\n|---|---|---|---|---|---|')
for p,h,k,s,pu,rho in rows: print(f"| {p} | {'' if h is None else round(h,5)} | {k} | {s} | {pu or ''} | {rho} |")
