import numpy as np


def dynamics(f):
    def col(i):return f[f'feature{i}'].to_numpy().astype('d')
    now=f['feature1'].dt.epoch('ms').to_numpy().astype('d')/1000
    stamps=[col(57),col(58),col(59)/1000];stamps[0]=np.where(stamps[0]>1e11,stamps[0]/1000,stamps[0]);age=[]
    for stamp in stamps:
        days=(now-stamp)/86400;age.append(np.log1p(np.where((stamp>0)&(days>=0),days,np.nan)))
    lag=stamps[1]-stamps[2];age.append(np.where((stamps[1]>0)&(stamps[2]>0),np.sign(lag)*np.log1p(abs(lag)),np.nan));hour=(now%86400)/86400;age.extend([np.sin(2*np.pi*hour),np.cos(2*np.pi*hour)])
    den=np.column_stack([col(k) for k in range(189,193)]);hist=[]
    for start in (213,221):
        num=np.column_stack([col(k) for k in range(start,start+4)]);n=np.diff(den,axis=1);s=np.diff(num,axis=1);ok=(den>=0)&(num>=0)&(num<=den)&np.isfinite(den)&np.isfinite(num);pair=ok[:,1:]&ok[:,:-1]&(n>0)&(s>=0)&(s<=n)
        # Compare each disjoint increment's rate to the previous cumulative rate, with shrinkage.
        previous=(num[:,:-1]+1)/(np.maximum(den[:,:-1],0)+20);rate=(np.maximum(s,0)+20*previous)/(np.maximum(n,0)+20);hist.extend([np.where(pair,rate,np.nan),np.where(pair,rate-previous,np.nan)])
    return np.column_stack(age).astype('f4'),np.column_stack(hist).astype('f4')
