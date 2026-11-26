"""T090: Enriched row-only TabM k32 (w128/w256) and DCNv2-r32 on two chronological folds."""
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
from tabm import TabM

OUT = Path('/kaggle/working/t090')
CACHE = Path('/kaggle/temp/t090')
PRIMARY_CUT = np.datetime64('2026-10-28T08:06:52.397')
EARLIER_CUT = np.datetime64('2026-10-27T08:06:52.397')
EARLIER_END = np.datetime64('2026-10-27T18:00:00')
LOW_CATS = [f'feature{i}' for i in [6,7,8,9,10,11,12,13,14,16,17,19,21,22,25]]
HIST_DUR = [476,477,479,480,483,484,486,487,492,493,500,501,502,503,512,513,514,515]
VEC_COLS = [f'feature{i}' for i in [*range(26,33), *range(40,47)]]


def numeric_cols(schema):
    return ['feature3'] + [
        c for c, t in schema.items()
        if c.startswith('feature') and c not in ('feature1', 'feature3')
        and str(t) in ('Float32', 'Int64', 'Int8')
    ]


def row_features(f, num_cols):
    n = len(f)
    raw = np.column_stack([f[c].to_numpy().astype('f4') for c in num_cols])
    vecs = np.stack([f[c].fill_null([0.0]*32).list.to_array(32).to_numpy().astype('f4') for c in VEC_COLS], axis=1)
    norms = np.linalg.norm(vecs, axis=2)
    present = (norms > 1e-6).astype('f4')
    units = vecs / np.maximum(norms[:, :, None], 1e-6)
    vec_block = np.concatenate([units, np.log1p(norms)[:, :, None], present[:, :, None]], axis=2).reshape(n, 14*34)
    bias = np.column_stack([f[f'feature{i}'].to_numpy().astype('f4') for i in [*range(33,40), *range(47,54)]])
    dots = np.einsum('bik,bjk->bij', vecs[:, :7], vecs[:, 7:]) + bias[:, :7, None] + bias[:, None, 7:]
    cos49 = np.einsum('bik,bjk->bij', units[:, :7], units[:, 7:])
    d = f['feature54'].to_numpy().astype('d')
    d = np.where(d > 0, d, np.nan)
    s = np.column_stack([f[f'feature{i}'].to_numpy().astype('d') for i in HIST_DUR])
    s = np.where(s >= 0, s, np.nan)
    dur = np.column_stack([np.log1p(s) - np.log1p(d[:, None]), s - d[:, None]]).astype('f4')
    now = f['feature1'].dt.epoch('ms').to_numpy().astype('d') / 1000.0
    st57 = f['feature57'].to_numpy().astype('d')
    st57 = np.where(st57 > 1e11, st57 / 1000.0, st57)
    st58 = f['feature58'].to_numpy().astype('d')
    st59 = f['feature59'].to_numpy().astype('d') / 1000.0
    ages = [np.log1p(np.where((st > 0) & (now >= st), (now - st) / 86400.0, np.nan)) for st in (st57, st58, st59)]
    hr = (now % 86400.0) / 86400.0
    f352 = f['feature352'].to_numpy().astype('d')
    f358 = f['feature358'].to_numpy().astype('d')
    f346 = f['feature346'].to_numpy().astype('d')
    f418 = f['feature418'].to_numpy().astype('d')
    f419 = f['feature419'].to_numpy().astype('d')
    f430 = f['feature430'].to_numpy().astype('d')
    extra = np.column_stack([
        *ages, np.sin(2*np.pi*hr), np.cos(2*np.pi*hr),
        f352 * f418, f346 * f419,
        f352 / (np.maximum(f358, 0) + 1e-4),
        f418 / (np.maximum(f430, 0) + 1e-4),
        np.log1p(np.maximum(f['feature493'].to_numpy().astype('d'), 0)) - np.log1p(d),
        np.log1p(np.maximum(f['feature480'].to_numpy().astype('d'), 0)) - np.log1p(d),
    ]).astype('f4')
    dense = np.column_stack([raw, dots.reshape(n, 49), cos49.reshape(n, 49), dur, extra]).astype('f4')
    return dense, vec_block.astype('f4')


def prepare(raw_path):
    schema = pl.scan_parquet(raw_path).collect_schema()
    num_cols = numeric_cols(schema)
    assert 'request_id' not in num_cols and len(num_cols) == 584
    fields = ['feature1'] + num_cols + VEC_COLS
    pf = pq.ParquetFile(raw_path)
    n = pf.metadata.num_rows
    assert n == 1537638
    n_dense = 584 + 49 + 49 + 36 + 11
    xd = np.lib.format.open_memmap(CACHE/'dense.npy', mode='w+', dtype='f4', shape=(n, n_dense))
    xv = np.lib.format.open_memmap(CACHE/'vec.npy', mode='w+', dtype='f4', shape=(n, 14*34))
    off = 0
    for batch in pf.iter_batches(columns=fields, batch_size=8192):
        f = pl.from_arrow(batch)
        d, v = row_features(f, num_cols)
        if off == 0:
            d1, v1 = row_features(f.slice(0, 1), num_cols)
            np.testing.assert_allclose(d[:1], d1, atol=1e-5, equal_nan=True)
            np.testing.assert_allclose(v[:1], v1, atol=1e-5, equal_nan=True)
        xd[off:off+len(f)] = d
        xv[off:off+len(f)] = v
        off += len(f)
    assert off == n
    xd.flush(); xv.flush()
    meta = pl.read_parquet(raw_path, columns=['feature1', 'watched'])
    np.save(CACHE/'t.npy', meta['feature1'].to_numpy())
    np.save(CACHE/'y.npy', meta['watched'].to_numpy())
    print(f'Prepared dense {xd.shape} and vec {xv.shape}', flush=True)


def build_cats(raw_path, tr_idx):
    f = pl.read_parquet(raw_path, columns=LOW_CATS)
    out, cards = [], []
    for col in LOW_CATS:
        s = f[col]
        vc = s.gather(tr_idx).drop_nulls().value_counts(name='cnt')
        if len(vc) > 1000:
            out.append(np.zeros(len(s), dtype=np.int16))
            cards.append(1)
            continue
        tab = vc.filter(pl.col('cnt') >= 10).sort(['cnt', col], descending=[True, False]).select(col).with_row_index('code', offset=1)
        codes = s.to_frame().join(tab, on=col, how='left', maintain_order='left', validate='m:1')['code'].fill_null(0).to_numpy().astype(np.int16)
        out.append(codes)
        cards.append(len(tab) + 1)
    return np.column_stack(out), cards


class LowRankCross(nn.Module):
    def __init__(self, dim, rank=32):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x0, x):
        return x + x0 * (self.up(self.down(x)) + self.bias)


class EnrichedNet(nn.Module):
    def __init__(self, n_dense, n_vec, cards, arch='tabm', width=128, members=32, dropout=0.2, pos_rate=0.05):
        super().__init__()
        self.n_dense = n_dense
        self.arch = arch
        self.plr = PeriodicEmbeddings(n_dense, 8, n_frequencies=8, frequency_init_scale=0.01, activation=True, lite=True)
        dims = [min(16, max(2, int(round(1.6 * c**0.56)))) if c > 1 else 0 for c in cards]
        self.embs = nn.ModuleList(nn.Embedding(c, d, padding_idx=0) if d > 0 else None for c, d in zip(cards, dims))
        for e in self.embs:
            if e is not None:
                nn.init.normal_(e.weight, std=0.02)
                with torch.no_grad():
                    e.weight[0].zero_()
        pairs = torch.triu_indices(14, 14, offset=1)
        self.register_buffer('pairs', pairs)
        in_dim = n_dense * 8 + n_vec + pairs.shape[1] * 4 + sum(dims)
        if arch == 'tabm':
            self.backbone = TabM.make(n_num_features=in_dim, d_out=1, k=members, n_blocks=2, d_block=width, dropout=dropout)
            bias = self.backbone.output.bias
        elif arch == 'dcnv2':
            self.crosses = nn.ModuleList(LowRankCross(in_dim, 32) for _ in range(2))
            self.deep = nn.Sequential(
                nn.Linear(in_dim, width), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(width, width), nn.ReLU(), nn.Dropout(dropout),
            )
            self.out = nn.Linear(in_dim + width, 1)
            bias = self.out.bias
        else:
            raise ValueError(arch)
        with torch.no_grad():
            bias.fill_(float(np.log(pos_rate / (1.0 - pos_rate))))

    def forward(self, dense, vec, cat):
        v = vec.reshape(-1, 14, 34)
        u = v[:, :, :32]
        norms = v[:, :, 32].expm1()
        pres = v[:, :, 33]
        i, j = self.pairs
        cos = (u[:, i] * u[:, j]).sum(-1).clamp(-1, 1)
        pr = pres[:, i] * pres[:, j]
        dot = cos * norms[:, i] * norms[:, j]
        dist = (norms[:, i].square() + norms[:, j].square() - 2 * dot).clamp_min(0).sqrt()
        crossed = torch.cat([
            cos * pr,
            dot.sign() * dot.abs().log1p().clamp_max(20) * pr,
            dist.log1p().clamp_max(20) * pr,
            pr,
        ], dim=1)
        if self.training:
            cat = cat.masked_fill(torch.rand_like(cat, dtype=torch.float32) < 0.3, 0)
        parts = [self.plr(dense).flatten(1), vec, crossed]
        for idx, e in enumerate(self.embs):
            if e is not None:
                parts.append(e(cat[:, idx].long()))
        h = torch.cat(parts, dim=1)
        if self.arch == 'tabm':
            return self.backbone(h).squeeze(-1)
        c = h
        for layer in self.crosses:
            c = layer(h, c)
        return self.out(torch.cat([c, self.deep(h)], dim=1)).squeeze(-1)


def fit_model(raw_path, fold, arch, width, seed, epochs=10):
    torch.manual_seed(seed); np.random.seed(seed)
    torch.set_float32_matmul_precision('high')
    device = 'cuda'
    t = np.load(CACHE/'t.npy'); y = np.load(CACHE/'y.npy')
    xd = np.load(CACHE/'dense.npy', mmap_mode='r')
    xv = np.load(CACHE/'vec.npy', mmap_mode='r')
    cut = PRIMARY_CUT if fold == 'primary' else EARLIER_CUT
    tr = np.flatnonzero(t < cut)
    va = np.flatnonzero((t >= cut) & ((fold == 'primary') | (t < EARLIER_END)))
    assert (len(tr), len(va)) == ((1230110, 307528) if fold == 'primary' else (710330, 299969))
    cats, cards = build_cats(raw_path, tr)
    rng = np.random.default_rng(42)
    samp = xd[np.sort(rng.choice(tr, min(65536, len(tr)), replace=False))].astype('d')
    med = np.nan_to_num(np.nanmedian(samp, axis=0), nan=0.0)
    samp = np.where(np.isfinite(samp), samp, med)
    sd = samp.std(axis=0)
    sd = np.where(sd > 1e-6, sd, 1.0)
    med = med.astype('f4'); sd = sd.astype('f4')
    del samp

    def load_gpu(rows):
        td = torch.empty((len(rows), xd.shape[1]), device=device, dtype=torch.float16)
        tv = torch.empty((len(rows), xv.shape[1]), device=device, dtype=torch.float16)
        for i in range(0, len(rows), 8192):
            sl = rows[i:i+8192]
            a = xd[sl]
            a = np.where(np.isfinite(a), a, med)
            a = np.clip((a - med) / sd, -8.0, 8.0)
            td[i:i+len(sl)] = torch.from_numpy(a).to(device=device, dtype=torch.float16)
            tv[i:i+len(sl)] = torch.from_numpy(np.asarray(xv[sl])).to(device=device, dtype=torch.float16)
        tc = torch.from_numpy(cats[rows]).to(device)
        return td, tv, tc

    tr_d, tr_v, tr_c = load_gpu(tr)
    va_d, va_v, va_c = load_gpu(va)
    tr_y = torch.from_numpy(y[tr].astype('f4')).to(device)
    model = EnrichedNet(xd.shape[1], xv.shape[1], cards, arch=arch, width=width, members=32, dropout=0.2, pos_rate=float(y[tr].mean())).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    eval_eps = (4, 6, 8, 10) if arch == 'tabm' else (2, 4, 6)
    max_ep = max(eval_eps)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_ep, eta_min=5e-5)
    best_auc, best_ep = -1.0, 0
    for ep in range(1, max_ep + 1):
        model.train()
        order = torch.randperm(len(tr), device=device)
        for j in order.split(1024):
            opt.zero_grad(set_to_none=True)
            logits = model(tr_d[j].float(), tr_v[j].float(), tr_c[j])
            if arch == 'tabm':
                loss = F.binary_cross_entropy_with_logits(logits, tr_y[j].unsqueeze(1).expand_as(logits))
            else:
                loss = F.binary_cross_entropy_with_logits(logits, tr_y[j])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()
        if ep in eval_eps:
            model.eval()
            preds = []
            with torch.inference_mode():
                for i in range(0, len(va), 2048):
                    out = model(va_d[i:i+2048].float(), va_v[i:i+2048].float(), va_c[i:i+2048]).sigmoid()
                    if arch == 'tabm':
                        out = out.mean(dim=1)
                    preds.append(out.cpu().numpy())
                s_out = model(va_d[:1].float(), va_v[:1].float(), va_c[:1]).sigmoid()
                if arch == 'tabm':
                    s_out = s_out.mean(dim=1)
                np.testing.assert_allclose(preds[0][:1], s_out.cpu().numpy(), atol=5e-4, rtol=5e-4)
            p = np.concatenate(preds).astype('f4')
            auc = float(roc_auc_score(y[va], p))
            print(f'{fold} {arch}-w{width}-s{seed} ep={ep} auc={auc:.6f}', flush=True)
            np.savez_compressed(OUT/f'{fold}-{arch}-w{width}-s{seed}-ep{ep}.npz', row_ids=va, y_true=y[va], probabilities=p)
            if auc > best_auc:
                best_auc, best_ep = auc, ep
    del tr_d, tr_v, tr_c, va_d, va_v, va_c, model, opt
    gc.collect(); torch.cuda.empty_cache()
    return {'best_auc': best_auc, 'best_ep': best_ep}


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True, parents=True); CACHE.mkdir(exist_ok=True, parents=True)
    raw = next(p for p in Path('/kaggle/input').rglob('train.parquet') if (p.parent/'sample_submission.csv').is_file())
    args = sys.argv[1:]; t0 = time.monotonic()
    if args == ['prepare']:
        prepare(raw); res = {}
    else:
        res = fit_model(raw, args[0], args[1], int(args[2]), int(args[3]))
    res['seconds'] = time.monotonic() - t0
    (OUT/('-'.join(args)+'.json')).write_text(json.dumps(res) + '\n')
