import numpy as np
from sklearn.cluster import MiniBatchKMeans

def paired(x):
    # Three predeclared cross-field pairs from the original T035 recipe.
    for j,k in [(6,10),(4,13),(5,11)]:
        u=x[:,j];v=x[:,k];u=u/np.maximum(np.linalg.norm(u,axis=1,keepdims=True),1e-8);v=v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-8);yield np.column_stack([u,v,u*v]),np.any(u!=0,axis=1)&np.any(v!=0,axis=1)

def learn(x,tr,kind):
    rng=np.random.default_rng(42);sample=x[np.sort(rng.choice(tr,min(32768,len(tr)),replace=False))];state=[]
    for mat,ok in paired(sample):
        mat=mat[ok];assert len(mat)>=24
        c=MiniBatchKMeans(n_clusters=24,n_init=1,max_iter=50,batch_size=1024,random_state=42).fit(mat).cluster_centers_ if kind=='learned' else mat[rng.choice(len(mat),24,replace=False)];state.append(c.astype('f4'))
    return state

def apply(x,state):
    parts=[]
    for (mat,ok),c in zip(paired(x),state):
        dist=np.maximum((mat*mat).sum(1)[:,None]+(c*c).sum(1)[None,:]-2*(mat@c.T),0);dist[~ok]=np.nan;parts.append(dist)
    return np.column_stack(parts)
