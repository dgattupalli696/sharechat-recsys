"""T095: Train-only quantile-encoded Enriched DCNv2-MoE4 (2 seeds) + TabM k32 (2 seeds) + multi-seed rank:pairwise XGBoost."""
import gc, json, sys, time
from pathlib import Path
import numpy as np
import polars as pl
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from rtdl_num_embeddings import PeriodicEmbeddings
from tabm import TabM
import t092_worker as base92
import t094_worker as base94
from quantiles81 import fit_quantiles, transform as q_transform

OUT = Path('/kaggle/working/t095')
CACHE = base94.CACHE


class QuantileTabM(nn.Module):
    def __init__(self, n_dense, n_vec, cards, width=256, members=32, dropout=0.2, pos_rate=0.05):
        super().__init__()
        self.plr = PeriodicEmbeddings(n_dense, 8, n_frequencies=8, frequency_init_scale=0.01, activation=True, lite=True)
        dims = [min(16, max(2, int(round(1.6 * c**0.56)))) if c > 1 else 0 for c in cards]
        self.embs = nn.ModuleList(nn.Embedding(c, d, padding_idx=0) if d > 0 else None for c, d in zip(cards, dims))
        for e in self.embs:
            if e is not None:
                nn.init.normal_(e.weight, std=0.02)
                with torch.no_grad(): e.weight[0].zero_()
        pairs = torch.triu_indices(14, 14, offset=1)
        self.register_buffer('pairs', pairs)
        in_dim = n_dense * 8 + n_vec + pairs.shape[1] * 4 + sum(dims)
        self.backbone = TabM.make(n_num_features=in_dim, d_out=1, k=members, n_blocks=2, d_block=width, dropout=dropout)
        with torch.no_grad():
            self.backbone.output.bias.fill_(float(np.log(pos_rate / (1.0 - pos_rate))))

    def forward(self, dense, vec, cat):
        v = vec.reshape(-1, 14, 34)
        u = v[:, :, :32]; norms = v[:, :, 32].expm1(); pres = v[:, :, 33]
        i, j = self.pairs
        cos = (u[:, i] * u[:, j]).sum(-1).clamp(-1, 1)
        pr = pres[:, i] * pres[:, j]
        dot = cos * norms[:, i] * norms[:, j]
        dist = (norms[:, i].square() + norms[:, j].square() - 2 * dot).clamp_min(0).sqrt()
        crossed = torch.cat([cos * pr, dot.sign() * dot.abs().log1p().clamp_max(20) * pr, dist.log1p().clamp_max(20) * pr, pr], dim=1)
        if self.training:
            cat = cat.masked_fill(torch.rand_like(cat, dtype=torch.float32) < 0.3, 0)
        parts = [self.plr(dense).flatten(1), vec, crossed]
        for idx, e in enumerate(self.embs):
            if e is not None: parts.append(e(cat[:, idx].long()))
        return self.backbone(torch.cat(parts, dim=1)).squeeze(-1)


def fit_q_neural(raw_path, fold, arch, seed):
    torch.manual_seed(seed); np.random.seed(seed); torch.set_float32_matmul_precision('high')
    device = 'cuda'
    tr, va, y = base94.get_split(fold)
    xd = np.load(CACHE/'dense.npy', mmap_mode='r'); xv = np.load(CACHE/'vec.npy', mmap_mode='r')
    cats, cards = base94.build_cats(raw_path, tr)
    # Fit 33-knot train-only quantile transform on 65,536 training rows
    rng = np.random.default_rng(42)
    samp = xd[np.sort(rng.choice(tr, min(65536, len(tr)), replace=False))].astype('f4')
    knots = fit_quantiles(samp)
    del samp

    def load_gpu(rows):
        td = torch.empty((len(rows), xd.shape[1]), device=device, dtype=torch.float16)
        tv = torch.empty((len(rows), xv.shape[1]), device=device, dtype=torch.float16)
        for i in range(0, len(rows), 8192):
            sl = rows[i:i+8192]
            td[i:i+len(sl)] = torch.from_numpy(q_transform(np.asarray(xd[sl]), knots)).to(device=device, dtype=torch.float16)
            tv[i:i+len(sl)] = torch.from_numpy(np.asarray(xv[sl])).to(device=device, dtype=torch.float16)
        return td, tv, torch.from_numpy(cats[rows]).to(device)

    tr_d, tr_v, tr_c = load_gpu(tr); va_d, va_v, va_c = load_gpu(va)
    tr_y = torch.from_numpy(y[tr].astype('f4')).to(device)
    if arch == 'q_tabm':
        model = QuantileTabM(xd.shape[1], xv.shape[1], cards, width=256, members=32, dropout=0.2, pos_rate=float(y[tr].mean())).to(device)
        eval_eps = (4, 6, 8)
    elif arch == 'q_moe4':
        model = base94.DiverseNet(xd.shape[1], xv.shape[1], cards, kind='dcnv2_moe4', pos_rate=float(y[tr].mean())).to(device)
        eval_eps = (2, 3, 4)
    else:
        raise ValueError(arch)
    opt = torch.optim.AdamW(model.parameters(), lr=4.5e-4, weight_decay=0.015)
    max_ep = max(eval_eps)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_ep, eta_min=5e-5)
    res = {}
    for ep in range(1, max_ep + 1):
        model.train()
        for j in torch.randperm(len(tr), device=device).split(1024):
            opt.zero_grad(set_to_none=True)
            logits = model(tr_d[j].float(), tr_v[j].float(), tr_c[j])
            if arch == 'q_tabm':
                loss = F.binary_cross_entropy_with_logits(logits, tr_y[j].unsqueeze(1).expand_as(logits))
            else:
                loss = F.binary_cross_entropy_with_logits(logits, tr_y[j])
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
        sched.step()
        if ep in eval_eps:
            model.eval(); preds = []
            with torch.inference_mode():
                for i in range(0, len(va), 2048):
                    out = model(va_d[i:i+2048].float(), va_v[i:i+2048].float(), va_c[i:i+2048]).sigmoid()
                    if arch == 'q_tabm': out = out.mean(dim=1)
                    preds.append(out.cpu().numpy())
                s1 = model(va_d[:1].float(), va_v[:1].float(), va_c[:1]).sigmoid()
                if arch == 'q_tabm': s1 = s1.mean(dim=1)
                np.testing.assert_allclose(preds[0][:1], s1.cpu().numpy(), atol=5e-4, rtol=5e-4)
            p = np.concatenate(preds).astype('f4')
            auc = float(roc_auc_score(y[va], p))
            res[str(ep)] = auc
            print(f'{fold} {arch}-s{seed} ep={ep} auc={auc:.6f}', flush=True)
            np.savez_compressed(OUT/f'{fold}-{arch}-s{seed}-ep{ep}.npz', row_ids=va, y_true=y[va], probabilities=p)
    del tr_d, tr_v, tr_c, va_d, va_v, va_c, model, opt
    gc.collect(); torch.cuda.empty_cache()
    return res


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True, parents=True); CACHE.mkdir(exist_ok=True, parents=True)
    raw = next(p for p in Path('/kaggle/input').rglob('train.parquet') if (p.parent/'sample_submission.csv').is_file())
    args = sys.argv[1:]; t0 = time.monotonic()
    if args == ['prepare']:
        base94.prepare(raw); res = {}
    else:
        res = fit_q_neural(raw, args[0], args[1], int(args[2]))
    res['seconds'] = time.monotonic() - t0
    (OUT/('-'.join(args)+'.json')).write_text(json.dumps(res) + '\n')
