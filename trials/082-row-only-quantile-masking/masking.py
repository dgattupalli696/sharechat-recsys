import torch

def mask(x,kind):
    if kind=='control':return x
    if kind=='field':return x*(torch.rand_like(x)>=.12)
    if kind!='group':raise ValueError(kind)
    # Equal marginal mask probability, but correlated within adjacent fields.
    g=torch.rand((len(x),(x.shape[1]+15)//16),device=x.device)>=.12
    return x*g.repeat_interleave(16,dim=1)[:,:x.shape[1]]
