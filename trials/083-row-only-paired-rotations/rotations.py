import numpy as np
from base62 import PAIRS

def learn(x,tr,shuffled):
    rng=np.random.default_rng(42);sample=x[np.sort(rng.choice(tr,min(32768,len(tr)),replace=False))].astype('d');mu=sample.mean(0);s=sample-mu;trans=[]
    for j in range(14):
        cov=s[:,j].T@s[:,j]/len(s);v,r=np.linalg.eigh(cov);trans.append((r*(np.maximum(v,0)+max(np.mean(v)*.05,1e-7))**-.5)@r.T)
    h=np.stack([s[:,j]@trans[j] for j in range(14)],axis=1);perm=rng.permutation(len(h)) if shuffled else np.arange(len(h));maps=[]
    for u,i in PAIRS:
        a,b=u-26,i-40+7;l,_,rt=np.linalg.svd(h[:,a].T@h[perm,b]/len(h));maps.append((a,b,(trans[a]@l[:,:12]).astype('f4'),(trans[b]@rt.T[:,:12]).astype('f4')))
    return mu.astype('f4'),maps

def apply(x,state):
    mu,maps=state;out=[]
    for a,b,l,r in maps:
        u=(x[:,a]-mu[a])@l;v=(x[:,b]-mu[b])@r;prod=u*v;z=np.column_stack([prod[:,:4],prod.mean(1),prod.sum(1)/np.maximum(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1),1e-8)]);z[(~np.any(x[:,a],axis=1))|(~np.any(x[:,b],axis=1))]=np.nan;out.append(z)
    return np.column_stack(out)
