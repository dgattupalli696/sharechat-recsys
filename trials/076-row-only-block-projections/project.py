"""Train-only oblique numeric views. All inference transforms act on a single row."""
import numpy as np


def blocks(names):
    return [[i for i,c in enumerate(names) if lo<=int(c[7:])<=hi] for lo,hi in [(60,155),(156,251),(252,347),(348,443),(444,539),(540,700)]]


def learn(x,train,names,kind):
    rng=np.random.default_rng(42);rows=np.sort(rng.choice(train,min(32768,len(train)),replace=False));sample=x[rows];state=[]
    for cols in blocks(names):
        v=sample[:,cols].astype('d');v[~np.isfinite(v)]=np.nan;q=np.nan_to_num(np.nanpercentile(v,[1,50,99],axis=0));v=np.clip(np.where(np.isfinite(v),v,q[1]),q[0],q[2]);mean=v.mean(0);scale=v.std(0);scale[scale<1e-7]=1.;z=(v-mean)/scale;k=min(8,len(cols));
        if kind=='pca':
            _,vectors=np.linalg.eigh(z.T@z/len(z));proj=vectors[:,-k:][:,::-1]
        elif kind=='random':proj=np.linalg.qr(rng.normal(size=(len(cols),k)))[0]
        else:raise ValueError(kind)
        state.append((cols,q,mean,scale,proj))
    return state


def apply(x,state):
    parts=[]
    for cols,q,mean,scale,proj in state:
        v=x[:,cols].astype('d');z=(np.clip(np.where(np.isfinite(v),v,q[1]),q[0],q[2])-mean)/scale;p=z@proj;res=z-p@proj.T
        parts.extend([p,np.sqrt(np.mean(res*res,axis=1))[:,None],np.max(abs(res),axis=1)[:,None]])
    return np.column_stack(parts).astype('f4')
