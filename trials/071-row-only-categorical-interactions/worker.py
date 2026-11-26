"""Incremental categorical information on top of T070's numeric gated backbone."""
import json,sys,time
from pathlib import Path
import numpy as np
import polars as pl
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import t070

OUT=Path('/kaggle/working/t071');CACHE=t070.CACHE
VIEW=[6,7,8,9,10,11,12,13,14,16,17];CONTENT=[18,19,20,21,22,23,25];COLS=[f'feature{i}' for i in VIEW+CONTENT]

def vocabulary(s):
    # Fixed cap; deterministic tie break. Zero is reserved for unseen/missing values.
    c=s.drop_nulls().value_counts(name='n').filter(pl.col('n')>=10).sort(['n',s.name],descending=[True,False]).head(8191)
    return c.select(s.name).with_row_index('code',offset=1)

def codes(s,tab):return s.to_frame().join(tab,on=s.name,how='left',maintain_order='left',validate='m:1')['code'].fill_null(0).to_numpy().astype('int16')

class WithCategories(nn.Module):
    def __init__(self,width,cards,kind):
        super().__init__();self.numeric=t070.Model(width,'gated');self.kind=kind
        if kind!='control':
            self.tables=nn.ModuleList(nn.Embedding(n,8,padding_idx=0) for n in cards);dim=8*len(cards)+(len(VIEW)*len(CONTENT) if kind=='cross' else 0);self.extra=nn.Sequential(nn.Linear(dim,64),nn.SiLU(),nn.Dropout(.15),nn.Linear(64,1));nn.init.zeros_(self.extra[-1].bias)
    def forward(self,x,c):
        y=self.numeric(x)
        if self.kind=='control':return y
        if self.training:c=c.masked_fill(torch.rand(c.shape,device=c.device)<.12,0)
        e=torch.stack([tab(c[:,i].long()) for i,tab in enumerate(self.tables)],1);z=e.flatten(1)
        if self.kind=='cross':z=torch.cat([z,(torch.bmm(e[:,:len(VIEW)],e[:,len(VIEW):].transpose(1,2))/(8**.5)).flatten(1)],1)
        return y+self.extra(z).squeeze(-1)

def fit(raw,fold,kind):
    torch.manual_seed(43);torch.set_num_threads(2);torch.set_float32_matmul_precision('high');rng=np.random.default_rng(42);t=np.load(CACHE/'t.npy');x=np.load(CACHE/'x.npy',mmap_mode='r');y=np.load(CACHE/'y.npy');cut=t070.t069.base.CUTOFF if fold=='primary' else np.datetime64('2026-10-27T08:06:52.397');tr=np.flatnonzero(t<cut);va=np.flatnonzero((t>=cut)&((fold=='primary')|(t<np.datetime64('2026-10-27T18:00'))));assert (len(tr),len(va))==((1230110,307528) if fold=='primary' else (710330,299969));assert x.shape[1]==1141
    f=pl.read_parquet(raw,columns=COLS);cats=[];cards=[]
    for col in COLS:
        s=f[col];tab=vocabulary(s.gather(tr));cats.append(codes(s,tab));cards.append(len(tab)+1)
    c=np.column_stack(cats);del f,cats;sample=x[np.sort(rng.choice(tr,32768,replace=False))];q=np.nanpercentile(sample,[10,50,90],axis=0);loc=np.nan_to_num(q[1]).astype('f4');sc=np.nan_to_num((q[2]-q[0])/2.563).astype('f4');sd=np.nanstd(sample,axis=0);sc=np.where(sc>1e-6,sc,np.where(sd>1e-6,sd,1));del sample
    def cache(ix):
        xx=torch.empty((len(ix),x.shape[1]),device='cuda',dtype=torch.float16)
        for i in range(0,len(ix),4096):xx[i:i+4096]=torch.from_numpy(np.nan_to_num(np.clip((x[ix[i:i+4096]]-loc)/sc,-8,8),nan=0)).to(device='cuda',dtype=xx.dtype)
        return xx,torch.from_numpy(c[ix]).to('cuda')
    train,ct=cache(tr);valid,cv=cache(va);labels=torch.from_numpy(y[tr].astype('f4')).to('cuda');model=WithCategories(x.shape[1],cards,kind).cuda();nn.init.constant_(model.numeric.head.bias,float(np.log(y[tr].mean()/(1-y[tr].mean()))));opt=torch.optim.AdamW(model.parameters(),lr=.00045,weight_decay=.015)
    for epoch in range(1,9):
        model.train()
        for j in torch.randperm(len(tr),device='cuda').split(2048):
            opt.zero_grad(set_to_none=True);F.binary_cross_entropy_with_logits(model(train[j].float(),ct[j]),labels[j]).backward();nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
        if epoch==5:
            for group in opt.param_groups:group['lr']=.00015
        if epoch in [5,8]:
            model.eval()
            with torch.inference_mode():
                def pred(z,cat):return model(z.float(),cat).sigmoid().cpu().numpy()
                p=np.concatenate([pred(z,cat) for z,cat in zip(valid.split(2048),cv.split(2048))]);np.testing.assert_allclose(p[:12],np.concatenate([pred(valid[i:i+1],cv[i:i+1]) for i in range(12)]),atol=2e-4,rtol=2e-4)
            assert np.isfinite(p).all();np.savez_compressed(OUT/f'{fold}-{kind}-{epoch}.npz',row_ids=va,y_true=y[va],probabilities=p);print(fold,kind,epoch,roc_auc_score(y[va],p),flush=True)

if __name__=='__main__':
    OUT.mkdir(exist_ok=True,parents=True);CACHE.mkdir(exist_ok=True,parents=True);raw=next(p for p in Path('/kaggle/input').rglob('train.parquet') if (p.parent/'sample_submission.csv').exists());args=sys.argv[1:];start=time.monotonic()
    if args==['prepare']:t070.t069.prepare(raw)
    else:fit(raw,*args)
    (OUT/('-'.join(args)+'.json')).write_text(json.dumps({'seconds':time.monotonic()-start})+'\n')
