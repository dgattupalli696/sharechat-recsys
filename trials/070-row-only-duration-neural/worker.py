"""Same-row neural models on the exact T069 enriched input. Train-only scaling."""
from pathlib import Path
import json,sys,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
import t069

OUT=Path('/kaggle/working/t070');CACHE=t069.CACHE

class Model(nn.Module):
    def __init__(self,width,kind):
        super().__init__();self.kind=kind;self.project=nn.Sequential(nn.Linear(width,192),nn.LayerNorm(192),nn.SiLU());self.blocks=nn.ModuleList(nn.Sequential(nn.LayerNorm(192),nn.Linear(192,384),nn.SiLU(),nn.Dropout(.18),nn.Linear(384,192)) for _ in range(2));self.head=nn.Linear(192,1)
        if kind=='gated':self.gate=nn.Linear(192,192);self.value=nn.Linear(192,192)
    def forward(self,x):
        h=self.project(x);initial=h
        for block in self.blocks:h=h+.5*block(h)
        if self.kind=='gated':h=h+initial*torch.sigmoid(self.gate(h))*self.value(h)
        return self.head(h).squeeze(-1)

def fit(fold,kind):
    assert kind in ('residual','gated');torch.manual_seed(42);rng=np.random.default_rng(42);torch.set_num_threads(2);torch.set_float32_matmul_precision('high');device='cuda';t=np.load(CACHE/'t.npy');y=np.load(CACHE/'y.npy');x=np.load(CACHE/'x.npy',mmap_mode='r');cut=t069.base.CUTOFF if fold=='primary' else np.datetime64('2026-10-27T08:06:52.397');tr=np.flatnonzero(t<cut);va=np.flatnonzero((t>=cut)&((fold=='primary')|(t<np.datetime64('2026-10-27T18:00'))));assert (len(tr),len(va))==((1230110,307528) if fold=='primary' else (710330,299969));assert x.shape[1]==1141
    # Fit a robust location and scale on a fixed sample of training rows only.
    sample=x[np.sort(rng.choice(tr,32768,replace=False))];q=np.nanpercentile(sample,[10,50,90],axis=0);loc=np.nan_to_num(q[1]).astype('f4');scale=np.nan_to_num((q[2]-q[0])/2.563).astype('f4');sd=np.nanstd(sample,axis=0);scale=np.where(scale>1e-6,scale,np.where(sd>1e-6,sd,1)).astype('f4');del sample
    def encode(rows):
        z=(x[rows]-loc)/scale;return np.nan_to_num(np.clip(z,-8,8),nan=0).astype('f4',copy=False)
    # Cache only this fold on device; avoid repeated random reads of the memmap.
    def cache(rows):
        z=torch.empty((len(rows),x.shape[1]),device=device,dtype=torch.float16)
        for i in range(0,len(rows),4096):z[i:i+4096]=torch.from_numpy(encode(rows[i:i+4096])).to(device=device,dtype=z.dtype)
        return z
    train=cache(tr);valid=cache(va);labels=torch.from_numpy(y[tr].astype('f4')).to(device);model=Model(x.shape[1],kind).to(device);nn.init.constant_(model.head.bias,float(np.log(y[tr].mean()/(1-y[tr].mean()))));opt=torch.optim.AdamW(model.parameters(),lr=.00045,weight_decay=.015);metrics={}
    for epoch in range(1,9):
        model.train();order=torch.randperm(len(tr),device=device)
        for j in order.split(2048):
            opt.zero_grad(set_to_none=True);loss=F.binary_cross_entropy_with_logits(model(train[j].float()),labels[j]);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
        if epoch==5:
            for group in opt.param_groups:group['lr']=.00015
        if epoch in (2,5,8):
            model.eval()
            with torch.inference_mode():
                def predict(z):return model(z.float()).sigmoid().cpu().numpy()
                p=np.concatenate([predict(z) for z in valid.split(2048)]);np.testing.assert_allclose(p[:12],np.concatenate([predict(valid[i:i+1]) for i in range(12)]),atol=2e-4,rtol=2e-4)
            assert np.isfinite(p).all();auc=roc_auc_score(y[va],p);metrics[epoch]=auc;np.savez_compressed(OUT/f'{fold}-{kind}-{epoch}.npz',row_ids=va,y_true=y[va],probabilities=p);print(fold,kind,epoch,auc,flush=True)
    return metrics

if __name__=='__main__':
    args=sys.argv[1:];OUT.mkdir(exist_ok=True,parents=True);CACHE.mkdir(exist_ok=True,parents=True);start=time.monotonic()
    if args==['prepare']:t069.prepare(next(p for p in Path('/kaggle/input').rglob('train.parquet') if (p.parent/'sample_submission.csv').exists()));scores={}
    else:scores=fit(*args)
    (OUT/('-'.join(args)+'.json')).write_text(json.dumps({'scores':scores,'seconds':time.monotonic()-start})+'\n')
