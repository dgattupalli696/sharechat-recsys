"""Same-row neural models on the exact T069 enriched input. Train-only scaling."""
from pathlib import Path
import json,sys,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
import t069
from quantiles81 import fit_quantiles,transform

OUT=Path('/kaggle/working/t081');CACHE=t069.CACHE

class Model(nn.Module):
    def __init__(self,width,kind):
        super().__init__();self.kind=kind;self.project=nn.Sequential(nn.Linear(width,192),nn.LayerNorm(192),nn.SiLU());self.blocks=nn.ModuleList(nn.Sequential(nn.LayerNorm(192),nn.Linear(192,384),nn.SiLU(),nn.Dropout(.18),nn.Linear(384,192)) for _ in range(2));self.head=nn.Linear(192,1)
        if kind=='gated':self.gate=nn.Linear(192,192);self.value=nn.Linear(192,192)
    def forward(self,x):
        h=self.project(x);initial=h
        for block in self.blocks:h=h+.5*block(h)
        if self.kind=='gated':h=h+initial*torch.sigmoid(self.gate(h))*self.value(h)
        return self.head(h).squeeze(-1)

def fit(kind):
    from worker87 import N,V,CACHE,save
    assert kind in ('quantile','gated');torch.manual_seed(42);rng=np.random.default_rng(42);torch.set_num_threads(2);torch.set_float32_matmul_precision('high');device='cuda';y=np.load(CACHE/'y.npy');x=np.load(CACHE/'x.npy',mmap_mode='r');tr=np.arange(N);assert x.shape==(N+V,1141) and len(y)==N
    # Fit a robust location and scale on a fixed sample of training rows only.
    sample=x[np.sort(rng.choice(tr,32768,replace=False))];q=np.nanpercentile(sample,[10,50,90],axis=0);loc=np.nan_to_num(q[1]).astype('f4');scale=np.nan_to_num((q[2]-q[0])/2.563).astype('f4');sd=np.nanstd(sample,axis=0);scale=np.where(scale>1e-6,scale,np.where(sd>1e-6,sd,1)).astype('f4');knots=fit_quantiles(sample);del sample
    def encode(rows):
        z=(x[rows]-loc)/scale;robust=np.nan_to_num(np.clip(z,-8,8),nan=0).astype('f4',copy=False);return transform(x[rows],knots) if kind=='quantile' else robust
    # Cache only this fold on device; avoid repeated random reads of the memmap.
    def cache(rows):
        z=torch.empty((len(rows),x.shape[1]),device=device,dtype=torch.float16)
        for i in range(0,len(rows),4096):z[i:i+4096]=torch.from_numpy(encode(rows[i:i+4096])).to(device=device,dtype=z.dtype)
        return z
    train=cache(tr);labels=torch.from_numpy(y[tr].astype('f4')).to(device);model=Model(x.shape[1],'gated').to(device);nn.init.constant_(model.head.bias,float(np.log(y[tr].mean()/(1-y[tr].mean()))));opt=torch.optim.AdamW(model.parameters(),lr=.00045,weight_decay=.015);metrics={}
    for epoch in range(1,9):
        model.train();order=torch.randperm(len(tr),device=device)
        for j in order.split(2048):
            opt.zero_grad(set_to_none=True);loss=F.binary_cross_entropy_with_logits(model(train[j].float()),labels[j]);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
        if epoch==5:
            for group in opt.param_groups:group['lr']=.00015
    model.eval()
    with torch.inference_mode():
        def pred(i,j):return model(torch.from_numpy(encode(slice(i,j))).to(device,dtype=torch.float16).float()).sigmoid().cpu().numpy()
        p=np.concatenate([pred(i,min(len(x),i+2048)) for i in range(N,len(x),2048)]);np.testing.assert_allclose(p[:12],np.concatenate([pred(i,i+1) for i in range(N,N+12)]),atol=2e-4,rtol=2e-4);save(kind,p)
