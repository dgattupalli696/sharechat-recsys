"""T096: Full-data (1,537,638 rows) row-only refit of our best validated T092/T094/T095 components."""
import gc, json, sys, time
from pathlib import Path
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.nn import functional as F
import t092_worker as base92
import t094_worker as base94
import t095_worker as base95
from quantiles81 import fit_quantiles, transform as q_transform

N = 1537638
V = 1037090
OUT = Path('/kaggle/working/t096')
CACHE = Path('/kaggle/temp/t096')
LOW_CATS = base92.LOW_CATS
VEC_COLS = base92.VEC_COLS


def prepare(data_dir):
    schema = pl.scan_parquet(data_dir / 'train.parquet').collect_schema()
    num_cols = base92.numeric_cols(schema)
    assert 'request_id' not in num_cols and len(num_cols) == 584
    fields = ['feature1'] + num_cols + VEC_COLS
    keep_dense = np.r_[np.arange(0, 584), np.arange(584 + 448, 1240)]
    xf = np.lib.format.open_memmap(CACHE / 'full.npy', mode='w+', dtype='f4', shape=(N + V, 1240))
    xd = np.lib.format.open_memmap(CACHE / 'dense.npy', mode='w+', dtype='f4', shape=(N + V, 792))
    xv = np.lib.format.open_memmap(CACHE / 'vec.npy', mode='w+', dtype='f4', shape=(N + V, 14 * 34))
    off = 0
    for split, rows in [('train', N), ('test', V)]:
        pf = pq.ParquetFile(data_dir / f'{split}.parquet')
        assert pf.metadata.num_rows == rows
        for batch in pf.iter_batches(columns=fields, batch_size=8192):
            f = pl.from_arrow(batch)
            z = base92.super_features(f, num_cols)
            vecs = z[:, 584:584 + 448].reshape(len(f), 14, 32)
            norms = np.linalg.norm(vecs, axis=2)
            pres = (norms > 1e-6).astype('f4')
            units = vecs / np.maximum(norms[:, :, None], 1e-6)
            vb = np.concatenate([units, np.log1p(norms)[:, :, None], pres[:, :, None]], axis=2).reshape(len(f), 14 * 34)
            xf[off:off + len(f)] = z
            xd[off:off + len(f)] = z[:, keep_dense]
            xv[off:off + len(f)] = vb
            off += len(f)
    assert off == N + V
    xf.flush(); xd.flush(); xv.flush()
    y = pl.read_parquet(data_dir / 'train.parquet', columns=['watched'])['watched'].to_numpy()
    assert len(y) == N
    np.save(CACHE / 'y.npy', y)
    ids = pl.read_csv(data_dir / 'sample_submission.csv')['id']
    assert len(ids) == ids.n_unique() == V
    np.save(OUT / 'ids.npy', ids.to_numpy())
    print(f'Prepared full train+test caches ({N}+{V})', flush=True)


def save_pred(name, p):
    p = np.asarray(p, dtype=np.float32)
    assert p.shape == (V,) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    np.save(OUT / f'{name}.npy', p)


def fit_xgb_full(arm):
    import cupy as cp
    import xgboost as xgb
    x = np.load(CACHE / 'full.npy', mmap_mode='r')
    y = np.load(CACHE / 'y.npy')
    is_rank = (arm == 'rank_d5')
    if arm == 'xgb_d6':
        params = dict(objective='binary:logistic', tree_method='hist', device='cuda', max_depth=6, eta=0.02, subsample=0.8, colsample_bytree=0.4, colsample_bynode=0.5, min_child_weight=150, reg_lambda=20, reg_alpha=2, gamma=1, max_bin=127, seed=42, nthread=4)
        rounds = 1125  # 900 * 1.25 for 25% more training rows
    elif arm == 'xgb_d5':
        params = dict(objective='binary:logistic', tree_method='hist', device='cuda', max_depth=5, eta=0.02, subsample=0.8, colsample_bytree=0.35, colsample_bynode=0.5, min_child_weight=200, reg_lambda=30, reg_alpha=5, gamma=1, max_bin=127, seed=43, nthread=4)
        rounds = 2400
    elif arm == 'rank_d5':
        params = dict(objective='rank:pairwise', lambdarank_pair_method='mean', lambdarank_num_pair_per_sample=8, tree_method='hist', device='cuda', max_depth=5, eta=0.02, subsample=0.8, colsample_bytree=0.35, colsample_bynode=0.5, min_child_weight=200, reg_lambda=30, reg_alpha=5, gamma=1, max_bin=127, seed=51, nthread=4)
        rounds = 625  # 500 * 1.25
    else:
        raise ValueError(arm)

    class Iter(xgb.DataIter):
        def __init__(self): self.reset(); super().__init__()
        def reset(self): self.i = 0
        def next(self, input_data):
            if self.i >= N: return False
            i = self.i; j = min(N, i + 8192); self.i = j
            kw = dict(data=cp.asarray(x[i:j]), label=cp.asarray(y[i:j]))
            if is_rank: kw['qid'] = cp.zeros(j - i, dtype=cp.uint32)
            input_data(**kw)
            return True

    dm = xgb.QuantileDMatrix(Iter(), max_bin=127, nthread=4)
    model = xgb.train(params, dm, rounds)
    del dm; gc.collect(); cp.get_default_memory_pool().free_all_blocks()

    def pred(i, j):
        out = cp.asnumpy(model.inplace_predict(cp.asarray(x[i:j])))
        return (1.0 / (1.0 + np.exp(-np.clip(out, -20, 20)))) if is_rank else out

    p = np.concatenate([pred(i, min(N + V, i + 8192)) for i in range(N, N + V, 8192)])
    np.testing.assert_allclose(p[:8], np.concatenate([pred(i, i + 1) for i in range(N, N + 8)]), atol=1e-6)
    save_pred(arm, p)


def build_cats_full(data_dir):
    tr_f = pl.read_parquet(data_dir / 'train.parquet', columns=LOW_CATS)
    te_f = pl.read_parquet(data_dir / 'test.parquet', columns=LOW_CATS)
    tr_out, te_out, cards = [], [], []
    for col in LOW_CATS:
        s_tr, s_te = tr_f[col], te_f[col]
        vc = s_tr.drop_nulls().value_counts(name='cnt')
        if len(vc) > 1000:
            tr_out.append(np.zeros(N, dtype=np.int16))
            te_out.append(np.zeros(V, dtype=np.int16))
            cards.append(1)
            continue
        tab = vc.filter(pl.col('cnt') >= 10).sort(['cnt', col], descending=[True, False]).select(col).with_row_index('code', offset=1)
        tr_out.append(s_tr.to_frame().join(tab, on=col, how='left', maintain_order='left', validate='m:1')['code'].fill_null(0).to_numpy().astype(np.int16))
        te_out.append(s_te.to_frame().join(tab, on=col, how='left', maintain_order='left', validate='m:1')['code'].fill_null(0).to_numpy().astype(np.int16))
        cards.append(len(tab) + 1)
    return np.column_stack(tr_out), np.column_stack(te_out), cards


def fit_neural_full(data_dir, arm, seed=42):
    torch.manual_seed(seed); np.random.seed(seed); torch.set_float32_matmul_precision('high')
    device = 'cuda'
    xd = np.load(CACHE / 'dense.npy', mmap_mode='r')
    xv = np.load(CACHE / 'vec.npy', mmap_mode='r')
    y = np.load(CACHE / 'y.npy')
    tr_cat, te_cat, cards = build_cats_full(data_dir)
    rng = np.random.default_rng(42)
    samp = xd[np.sort(rng.choice(N, 65536, replace=False))]
    use_q = arm.startswith('q_')
    if use_q:
        knots = fit_quantiles(samp.astype('f4'))
        def enc(a): return q_transform(np.asarray(a), knots)
    else:
        s_d = samp.astype('d')
        med = np.nan_to_num(np.nanmedian(s_d, axis=0), nan=0.0)
        s_d = np.where(np.isfinite(s_d), s_d, med)
        sd = np.where(s_d.std(axis=0) > 1e-6, s_d.std(axis=0), 1.0)
        med = med.astype('f4'); sd = sd.astype('f4')
        def enc(a):
            a = np.where(np.isfinite(a), a, med)
            return np.clip((a - med) / sd, -8.0, 8.0)
    del samp

    tr_d = torch.empty((N, 792), device=device, dtype=torch.float16)
    tr_v = torch.empty((N, 14 * 34), device=device, dtype=torch.float16)
    for i in range(0, N, 8192):
        j = min(N, i + 8192)
        tr_d[i:j] = torch.from_numpy(enc(xd[i:j])).to(device=device, dtype=torch.float16)
        tr_v[i:j] = torch.from_numpy(np.asarray(xv[i:j])).to(device=device, dtype=torch.float16)
    tr_c = torch.from_numpy(tr_cat).to(device)
    tr_y = torch.from_numpy(y.astype('f4')).to(device)

    pos_rate = float(y.mean())
    if arm == 'dcnv2_r64':
        model = base94.DiverseNet(792, 14 * 34, cards, kind='dcnv2_r64', pos_rate=pos_rate).to(device)
        epochs, lr = 4, 4e-4
    elif arm in ('dcnv2_moe4', 'q_moe4'):
        model = base94.DiverseNet(792, 14 * 34, cards, kind='dcnv2_moe4', pos_rate=pos_rate).to(device)
        epochs, lr = 3, 4e-4
    elif arm == 'q_tabm':
        model = base95.QuantileTabM(792, 14 * 34, cards, width=256, members=32, dropout=0.2, pos_rate=pos_rate).to(device)
        epochs, lr = 8, 4.5e-4
    else:
        raise ValueError(arm)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.015)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=5e-5)
    for ep in range(1, epochs + 1):
        model.train()
        for j in torch.randperm(N, device=device).split(1024):
            opt.zero_grad(set_to_none=True)
            logits = model(tr_d[j].float(), tr_v[j].float(), tr_c[j])
            if arm == 'q_tabm':
                loss = F.binary_cross_entropy_with_logits(logits, tr_y[j].unsqueeze(1).expand_as(logits))
            else:
                loss = F.binary_cross_entropy_with_logits(logits, tr_y[j])
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
        sched.step()
    del tr_d, tr_v, tr_c, tr_y; gc.collect(); torch.cuda.empty_cache()

    model.eval(); preds = []
    with torch.inference_mode():
        def pred_slice(i, j):
            d_t = torch.from_numpy(enc(xd[N + i:N + j])).to(device=device, dtype=torch.float16).float()
            v_t = torch.from_numpy(np.asarray(xv[N + i:N + j])).to(device=device, dtype=torch.float16).float()
            c_t = torch.from_numpy(te_cat[i:j]).to(device)
            out = model(d_t, v_t, c_t).sigmoid()
            if arm == 'q_tabm': out = out.mean(dim=1)
            return out.cpu().numpy()
        for i in range(0, V, 2048):
            preds.append(pred_slice(i, min(V, i + 2048)))
        np.testing.assert_allclose(preds[0][:1], pred_slice(0, 1), atol=5e-4, rtol=5e-4)
    save_pred(f'{arm}_s{seed}', np.concatenate(preds))
    del model, opt; gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True, parents=True); CACHE.mkdir(exist_ok=True, parents=True)
    data_dir = next(p.parent for p in Path('/kaggle/input').rglob('train.parquet') if (p.parent / 'sample_submission.csv').is_file())
    args = sys.argv[1:]; t0 = time.monotonic()
    if args == ['prepare']:
        prepare(data_dir)
    elif args[0] in ('xgb_d6', 'xgb_d5', 'rank_d5'):
        fit_xgb_full(args[0])
    else:
        fit_neural_full(data_dir, args[0], int(args[1]) if len(args) > 1 else 42)
    (OUT / ('-'.join(args) + '.json')).write_text(json.dumps({'stage': args, 'fit': N, 'test': V, 'seconds': time.monotonic() - t0}) + '\n')
    print('DONE', args, f'{time.monotonic()-t0:.0f}s', flush=True)
