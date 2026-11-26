import numpy as np

def fit_quantiles(sample):
    levels=np.linspace(.005,.995,33);qs=np.nanquantile(sample,levels,axis=0);knots=[]
    for col in qs.T:
        ok=np.isfinite(col);unique,inv=np.unique(col[ok],return_inverse=True);values=np.bincount(inv,weights=levels[ok])/np.maximum(np.bincount(inv),1);knots.append((unique,values))
    return knots

def transform(x,knots):
    v=np.zeros(x.shape,dtype='f4')
    for i,(u,r) in enumerate(knots):
        if len(u)>1:v[:,i]=np.where(np.isfinite(x[:,i]),2*np.interp(x[:,i],u,r)-1,0.)
    return v
