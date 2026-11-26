"""T094: Enriched DCNv2-r64, DCNv2-MoE4, FinalMLP-gated + early-peaking global rank:pairwise XGBoost."""
import gc, json, sys, time
from pathlib import Path
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from rtdl_num_embeddings import PeriodicEmbeddings
import t092_worker as base92

OUT = Path('/kaggle/working/t094')
CACHE = Path('/kaggle/temp/t094')
LOW_CATS = base92.LOW_CATS
VEC_COLS = base92.VEC_COLS


def prepare(raw_path):
    # Reuse base92's 1,240 super-enriched features + separate 14x34 unit-vector block for pairwise vector crosses
    schema = pl.scan_parquet(raw_path).collect_schema()
    num_cols = base92.numeric_cols(schema)
    fields = ['feature1'] + num_cols + VEC_COLS
    pf = pq.ParquetFile(raw_path)
    n = pf.metadata.num_rows
    assert n == 1537638
    # Dense for neural: 1,240 super features minus the 448 raw vector coords (448 are in vec_block) = 792 dense + 476 vec
    xd = np.lib.format.open_memmap(CACHE/'dense.npy', mode='w+', dtype='f4', shape=(n, 792))
    xv = np.lib.format.open_memmap(CACHE/'vec.npy', mode='w+', dtype='f4', shape=(n, 14*34))
    xf = np.lib.format.open_memmap(CACHE/'full.npy', mode='w+', dtype='f4', shape=(n, 1240))
    keep_dense = np.r_[np.arange(0, 584), np.arange(584 + 448, 1240)]
    assert len(keep_dense) == 792
    off = 0
    for batch in pf.iter_batches(columns=fields, batch_size=8192):
        f = pl.from_arrow(batch)
        z = base92.super_features(f, num_cols)
        vecs = z[:, 584:584+448].reshape(len(f), 14, 32)
        norms = np.linalg.norm(vecs, axis=2)
        pres = (norms > 1e-6).astype('f4')
        units = vecs / np.maximum(norms[:, :, None], 1e-6)
        vb = np.concatenate([units, np.log1p(norms)[:, :, None], pres[:, :, None]], axis=2).reshape(len(f), 14*34)
        xf[off:off+len(f)] = z
        xd[off:off+len(f)] = z[:, keep_dense]
        xv[off:off+len(f)] = vb
        off += len(f)
    assert off == n
    xf.flush(); xd.flush(); xv.flush()
    meta = pl.read_parquet(raw_path, columns=['feature1', 'watched'])
    np.save(CACHE/'t.npy', meta['feature1'].to_numpy())
    np.save(CACHE/'y.npy', meta['watched'].to_numpy())
    print(f'Prepared T094 caches: full={xf.shape}, dense={xd.shape}, vec={xv.shape}', flush=True)


def get_split(fold):
    t = np.load(CACHE/'t.npy'); y = np.load(CACHE/'y.npy')
    cut = base92.PRIMARY_CUT if fold == 'primary' else base92.EARLIER_CUT
    tr = np.flatnonzero(t < cut)
    va = np.flatnonzero((t >= cut) & ((fold == 'primary') | (t < base92.EARLIER_END)))
    assert (len(tr), len(va)) == ((1230110, 307528) if fold == 'primary' else (710330, 299969))
    return tr, va, y


def build_cats(raw_path, tr_idx):
    f = pl.read_parquet(raw_path, columns=LOW_CATS)
    out, cards = [], []
    for col in LOW_CATS:
        s = f[col]
        vc = s.gather(tr_idx).drop_nulls().value_counts(name='cnt')
        if len(vc) > 1000:
            out.append(np.zeros(len(s), dtype=np.int16)); cards.append(1); continue
        tab = vc.filter(pl.col('cnt') >= 10).sort(['cnt', col], descending=[True, False]).select(col).with_row_index('code', offset=1)
        out.append(s.to_frame().join(tab, on=col, how='left', maintain_order='left', validate='m:1')['code'].fill_null(0).to_numpy().astype(np.int16))
        cards.append(len(tab) + 1)
    return np.column_stack(out), cards


class LowRankCross(nn.Module):
    def __init__(self, dim, rank=32, experts=1):
        super().__init__()
        self.experts = experts
        self.down = nn.ModuleList(nn.Linear(dim, rank, bias=False) for _ in range(experts))
        self.up = nn.ModuleList(nn.Linear(rank, dim, bias=False) for _ in range(experts))
        self.bias = nn.Parameter(torch.zeros(experts, dim))
        if experts > 1:
            self.mid = nn.ModuleList(nn.Linear(rank, rank, bias=False) for _ in range(experts))
            self.gate = nn.Linear(dim, experts, bias=False)

    def forward(self, x0, x):
        if self.experts == 1:
            return x + x0 * (self.up[0](self.down[0](x)) + self.bias[0])
        w = self.gate(x).softmax(dim=-1)
        out = [x0 * (self.up[i](self.mid[i](self.down[i](x).tanh()).tanh()) + self.bias[i]) for i in range(self.experts)]
        return x + (torch.stack(out, dim=1) * w.unsqueeze(-1)).sum(dim=1)


class DiverseNet(nn.Module):
    def __init__(self, n_dense, n_vec, cards, kind, pos_rate=0.05):
        super().__init__()
        self.kind = kind
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
        if kind == 'dcnv2_r64':
            self.crosses = nn.ModuleList(LowRankCross(in_dim, rank=64, experts=1) for _ in range(3))
            self.deep = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.2))
            self.out = nn.Linear(in_dim + 256, 1)
            bias = self.out.bias
        elif kind == 'dcnv2_moe4':
            self.crosses = nn.ModuleList(LowRankCross(in_dim, rank=32, experts=4) for _ in range(2))
            self.deep = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.2))
            self.out = nn.Linear(in_dim + 256, 1)
            bias = self.out.bias
        elif kind == 'finalmlp':
            self.streams = nn.ModuleList(
                nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.2))
                for _ in range(2)
            )
            self.gates = nn.ModuleList(
                nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, in_dim), nn.Sigmoid())
                for _ in range(2)
            )
            self.lin = nn.ModuleList(nn.Linear(256, 1, bias=False) for _ in range(2))
            self.bilin = nn.Parameter(torch.empty(4, 64, 64))
            for mat in self.bilin: nn.init.xavier_uniform_(mat)
            self.bias = nn.Parameter(torch.zeros(1))
            bias = self.bias
        else:
            raise ValueError(kind)
        with torch.no_grad():
            bias.fill_(float(np.log(pos_rate / (1.0 - pos_rate))))

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
        h = torch.cat(parts, dim=1)
        if self.kind in ('dcnv2_r64', 'dcnv2_moe4'):
            c = h
            for layer in self.crosses: c = layer(h, c)
            return self.out(torch.cat([c, self.deep(h)], dim=1)).squeeze(-1)
        s0 = self.streams[0](h * (2 * self.gates[0](h)))
        s1 = self.streams[1](h * (2 * self.gates[1](h)))
        inter = torch.einsum('bhd,hde,bhe->b', s0.reshape(-1, 4, 64), self.bilin, s1.reshape(-1, 4, 64))
        return (self.lin[0](s0) + self.lin[1](s1)).squeeze(-1) + inter + self.bias


def fit_neural(raw_path, fold, kind, seed=42):
    torch.manual_seed(seed); np.random.seed(seed); torch.set_float32_matmul_precision('high')
    device = 'cuda'
    tr, va, y = get_split(fold)
    xd = np.load(CACHE/'dense.npy', mmap_mode='r'); xv = np.load(CACHE/'vec.npy', mmap_mode='r')
    cats, cards = build_cats(raw_path, tr)
    rng = np.random.default_rng(42)
    samp = xd[np.sort(rng.choice(tr, min(65536, len(tr)), replace=False))].astype('d')
    med = np.nan_to_num(np.nanmedian(samp, axis=0), nan=0.0)
    samp = np.where(np.isfinite(samp), samp, med)
    sd = np.where(samp.std(axis=0) > 1e-6, samp.std(axis=0), 1.0)
    med = med.astype('f4'); sd = sd.astype('f4'); del samp

    def load_gpu(rows):
        td = torch.empty((len(rows), xd.shape[1]), device=device, dtype=torch.float16)
        tv = torch.empty((len(rows), xv.shape[1]), device=device, dtype=torch.float16)
        for i in range(0, len(rows), 8192):
            sl = rows[i:i+8192]
            a = xd[sl]; a = np.where(np.isfinite(a), a, med); a = np.clip((a - med) / sd, -8.0, 8.0)
            td[i:i+len(sl)] = torch.from_numpy(a).to(device=device, dtype=torch.float16)
            tv[i:i+len(sl)] = torch.from_numpy(np.asarray(xv[sl])).to(device=device, dtype=torch.float16)
        return td, tv, torch.from_numpy(cats[rows]).to(device)

    tr_d, tr_v, tr_c = load_gpu(tr); va_d, va_v, va_c = load_gpu(va)
    tr_y = torch.from_numpy(y[tr].astype('f4')).to(device)
    model = DiverseNet(xd.shape[1], xv.shape[1], cards, kind=kind, pos_rate=float(y[tr].mean())).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=0.015)
    eval_eps = (1, 2, 3, 4) if kind.startswith('dcn') else (2, 4, 6)
    max_ep = max(eval_eps)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_ep, eta_min=5e-5)
    res = {}
    for ep in range(1, max_ep + 1):
        model.train()
        for j in torch.randperm(len(tr), device=device).split(1024):
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(tr_d[j].float(), tr_v[j].float(), tr_c[j]), tr_y[j])
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
        sched.step()
        if ep in eval_eps:
            model.eval(); preds = []
            with torch.inference_mode():
                for i in range(0, len(va), 2048):
                    preds.append(model(va_d[i:i+2048].float(), va_v[i:i+2048].float(), va_c[i:i+2048]).sigmoid().cpu().numpy())
                s1 = model(va_d[:1].float(), va_v[:1].float(), va_c[:1]).sigmoid().cpu().numpy()
                np.testing.assert_allclose(preds[0][:1], s1, atol=5e-4, rtol=5e-4)
            p = np.concatenate(preds).astype('f4')
            auc = float(roc_auc_score(y[va], p))
            res[str(ep)] = auc
            print(f'{fold} {kind} ep={ep} auc={auc:.6f}', flush=True)
            np.savez_compressed(OUT/f'{fold}-{kind}-ep{ep}.npz', row_ids=va, y_true=y[va], probabilities=p)
    del tr_d, tr_v, tr_c, va_d, va_v, va_c, model, opt
    gc.collect(); torch.cuda.empty_cache()
    return res


def fit_rank_xgb(fold, arm):
    import cupy as cp
    import xgboost as xgb
    tr, va, y = get_split(fold)
    x = np.load(CACHE/'full.npy', mmap_mode='r')
    depth, seed = (5, 51) if arm == 'rank_d5' else (6, 52)
    params = dict(
        objective='rank:pairwise', lambdarank_pair_method='mean', lambdarank_num_pair_per_sample=8,
        tree_method='hist', device='cuda', max_depth=depth, eta=0.02, subsample=0.8,
        colsample_bytree=0.35, colsample_bynode=0.5, min_child_weight=200,
        reg_lambda=30, reg_alpha=5, gamma=1, max_bin=127, seed=seed, nthread=4,
    )
    rounds_list = (300, 500, 700, 900)

    class Iter(xgb.DataIter):
        def __init__(self): self.reset(); super().__init__()
        def reset(self): self.i = 0
        def next(self, input_data):
            if self.i >= len(tr): return False
            j = tr[self.i:self.i+8192]; self.i += 8192
            input_data(data=cp.asarray(x[j]), label=cp.asarray(y[j]), qid=cp.zeros(len(j), dtype=cp.uint32))
            return True

    dm = xgb.QuantileDMatrix(Iter(), max_bin=127, nthread=4)
    model = xgb.train(params, dm, max(rounds_list))
    del dm; gc.collect(); cp.get_default_memory_pool().free_all_blocks()
    res = {}
    for r in rounds_list:
        def pred(j):
            out = cp.asnumpy(model.inplace_predict(cp.asarray(x[j]), iteration_range=(0, r)))
            return 1.0 / (1.0 + np.exp(-np.clip(out, -20, 20)))
        p = np.concatenate([pred(va[i:i+8192]) for i in range(0, len(va), 8192)]).astype('f4')
        np.testing.assert_allclose(p[:8], np.concatenate([pred(j[None]) for j in va[:8]]), atol=1e-6)
        auc = float(roc_auc_score(y[va], p))
        res[str(r)] = auc
        print(f'{fold} {arm} r={r} auc={auc:.6f}', flush=True)
        np.savez_compressed(OUT/f'{fold}-{arm}-r{r}.npz', row_ids=va, y_true=y[va], probabilities=p)
    return res


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True, parents=True); CACHE.mkdir(exist_ok=True, parents=True)
    raw = next(p for p in Path('/kaggle/input').rglob('train.parquet') if (p.parent/'sample_submission.csv').is_file())
    args = sys.argv[1:]; t0 = time.monotonic()
    if args == ['prepare']:
        prepare(raw); res = {}
    elif args[1].startswith('rank_'):
        res = fit_rank_xgb(args[0], args[1])
    else:
        res = fit_neural(raw, args[0], args[1])
    res['seconds'] = time.monotonic() - t0
    (OUT/('-'.join(args)+'.json')).write_text(json.dumps(res) + '\n')
