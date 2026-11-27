"""T098: Export all trained model artifacts (.ubj, .pt, and frozen preprocessors) for S025 (T062+T066+T087+T096)."""
import gc, json, sys, time
from pathlib import Path
import joblib
import numpy as np
import polars as pl
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.nn import functional as F
import base62, rich63, t069, rot83, land86
import t092_worker as base92
import t094_worker as base94
import t095_worker as base95
from quantiles81 import fit_quantiles, transform as q_transform

N = 1537638
V = 1037090
OUT = Path('/kaggle/working/s025_artifacts')
CACHE = Path('/kaggle/temp/t098')
LOW_CATS = base92.LOW_CATS
VEC_COLS = base92.VEC_COLS


def prepare(data_dir):
    schema = pl.scan_parquet(data_dir / 'train.parquet').collect_schema()
    num_cols = base92.numeric_cols(schema)
    assert 'request_id' not in num_cols and len(num_cols) == 584
    fields = ['feature1'] + num_cols + VEC_COLS
    keep_dense = np.r_[np.arange(0, 584), np.arange(584 + 448, 1240)]
    # Cache 1: 1141 features (T062/T066/T087 layout: 594 base + 511 rich + 36 duration)
    x87 = np.lib.format.open_memmap(CACHE / 'x87.npy', mode='w+', dtype='f4', shape=(N + V, 1141))
    # Cache 2: 1240 features (T092/T096 layout) + 792 dense + 476 vec
    x96 = np.lib.format.open_memmap(CACHE / 'x96.npy', mode='w+', dtype='f4', shape=(N + V, 1240))
    xd = np.lib.format.open_memmap(CACHE / 'dense96.npy', mode='w+', dtype='f4', shape=(N + V, 792))
    xv = np.lib.format.open_memmap(CACHE / 'vec96.npy', mode='w+', dtype='f4', shape=(N + V, 14 * 34))
    off = 0
    for split, rows in [('train', N), ('test', V)]:
        pf = pq.ParquetFile(data_dir / f'{split}.parquet')
        assert pf.metadata.num_rows == rows
        for batch in pf.iter_batches(columns=fields, batch_size=8192):
            f = pl.from_arrow(batch)
            z87 = np.column_stack([base62.transform(f, num_cols), rich63.rich(f), t069.duration(f)])
            z96 = base92.super_features(f, num_cols)
            vecs = z96[:, 584:584 + 448].reshape(len(f), 14, 32)
            norms = np.linalg.norm(vecs, axis=2)
            pres = (norms > 1e-6).astype('f4')
            units = vecs / np.maximum(norms[:, :, None], 1e-6)
            vb = np.concatenate([units, np.log1p(norms)[:, :, None], pres[:, :, None]], axis=2).reshape(len(f), 14 * 34)
            x87[off:off + len(f)] = z87
            x96[off:off + len(f)] = z96
            xd[off:off + len(f)] = z96[:, keep_dense]
            xv[off:off + len(f)] = vb
            off += len(f)
    assert off == N + V
    x87.flush(); x96.flush(); xd.flush(); xv.flush()
    y = pl.read_parquet(data_dir / 'train.parquet', columns=['watched'])['watched'].to_numpy()
    np.save(CACHE / 'y.npy', y)
    ids = pl.read_csv(data_dir / 'sample_submission.csv')['id'].to_numpy()
    np.save(OUT / 'ids.npy', ids)
    print(f'Prepared caches for all 15 S025 tree/neural models ({N}+{V})', flush=True)


def save_pred(name, p):
    p = np.asarray(p, dtype=np.float32)
    assert p.shape == (V,) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    np.save(OUT / 'preds' / f'{name}.npy', p)


def compact87(z):
    return np.column_stack([z[:, :594], z[:, 1105:], z[:, 1042:1091]])


def export_xgb_87(arm):
    import cupy as cp
    import xgboost as xgb
    x = np.load(CACHE / 'x87.npy', mmap_mode='r')
    y = np.load(CACHE / 'y.npy')
    vec = x[:, 594:1042].reshape(len(x), 14, 32)
    tr = np.arange(N)

    state = None
    if arm == 't087_paired':
        state = rot83.learn(vec, tr, False)
        joblib.dump(state, OUT / 'models' / 't087_paired_rotations.joblib')
    elif arm in ('t087_learned', 't087_random'):
        kind = arm.split('_')[1]
        state = land86.learn(vec, tr, kind)
        joblib.dump(state, OUT / 'models' / f'{arm}_landmarks.joblib')

    def feat(i, j):
        z = x[i:j]
        if arm in ('t062_d12_r0', 't066_d6_r0'):
            return np.ascontiguousarray(z[:, :594])
        if arm in ('t066_d12_r1', 't066_d6_r1'):
            return np.ascontiguousarray(z[:, :1105])
        if arm == 't087_duration':
            return np.ascontiguousarray(z)
        if arm == 't087_compact':
            return np.ascontiguousarray(compact87(z))
        if arm == 't087_paired':
            return np.ascontiguousarray(np.column_stack([compact87(z), rot83.apply(vec[i:j], state)]))
        return np.ascontiguousarray(np.column_stack([compact87(z), land86.apply(vec[i:j], state)]))

    if arm in ('t062_d12_r0', 't066_d12_r1'):
        params = dict(objective='binary:logistic', tree_method='hist', device='cuda', max_depth=12, eta=0.02, subsample=0.8, reg_lambda=10, seed=42, max_bin=127, colsample_bytree=0.8, colsample_bynode=0.4, min_child_weight=100, gamma=1, nthread=4)
        rounds = 900
    elif arm in ('t066_d6_r0', 't066_d6_r1'):
        params = dict(objective='binary:logistic', tree_method='hist', device='cuda', max_depth=6, eta=0.02, subsample=0.8, reg_lambda=10, seed=42, max_bin=127, colsample_bytree=0.8, colsample_bynode=0.4, min_child_weight=100, gamma=1, nthread=4)
        rounds = 900
    elif arm == 't087_duration':
        params = dict(objective='binary:logistic', tree_method='hist', device='cuda', max_depth=6, eta=0.02, subsample=0.8, reg_lambda=10, seed=42, max_bin=127, colsample_bytree=0.8, colsample_bynode=0.4, min_child_weight=100, gamma=1, nthread=4)
        rounds = 2400
    else:
        params = dict(objective='binary:logistic', tree_method='hist', device='cuda', max_depth=8, eta=0.025, subsample=0.8, reg_lambda=10, seed=42, max_bin=127, colsample_bytree=0.65, colsample_bynode=0.5, min_child_weight=100, gamma=1, nthread=4)
        rounds = 1600

    class Iter(xgb.DataIter):
        def __init__(self): self.reset(); super().__init__()
        def reset(self): self.i = 0
        def next(self, input_data):
            if self.i >= N: return False
            i = self.i; j = min(N, i + 8192); self.i = j
            input_data(data=cp.asarray(feat(i, j)), label=cp.asarray(y[i:j]))
            return True

    dm = xgb.QuantileDMatrix(Iter(), max_bin=127, nthread=4)
    model = xgb.train(params, dm, rounds)
    model.save_model(str(OUT / 'models' / f'{arm}.ubj'))
    del dm; gc.collect(); cp.get_default_memory_pool().free_all_blocks()
    p = np.concatenate([cp.asnumpy(model.inplace_predict(cp.asarray(feat(i, min(N + V, i + 8192))))) for i in range(N, N + V, 8192)])
    save_pred(arm, p)


def export_xgb_96():
    import cupy as cp
    import xgboost as xgb
    x = np.load(CACHE / 'x96.npy', mmap_mode='r')
    y = np.load(CACHE / 'y.npy')
    params = dict(objective='binary:logistic', tree_method='hist', device='cuda', max_depth=6, eta=0.02, subsample=0.8, colsample_bytree=0.4, colsample_bynode=0.5, min_child_weight=150, reg_lambda=20, reg_alpha=2, gamma=1, max_bin=127, seed=42, nthread=4)

    class Iter(xgb.DataIter):
        def __init__(self): self.reset(); super().__init__()
        def reset(self): self.i = 0
        def next(self, input_data):
            if self.i >= N: return False
            i = self.i; j = min(N, i + 8192); self.i = j
            input_data(data=cp.asarray(x[i:j]), label=cp.asarray(y[i:j]))
            return True

    dm = xgb.QuantileDMatrix(Iter(), max_bin=127, nthread=4)
    model = xgb.train(params, dm, 1125)
    model.save_model(str(OUT / 'models' / 't096_xgb_d6.ubj'))
    del dm; gc.collect(); cp.get_default_memory_pool().free_all_blocks()
    p = np.concatenate([cp.asnumpy(model.inplace_predict(cp.asarray(x[i:min(N + V, i + 8192)]))) for i in range(N, N + V, 8192)])
    save_pred('t096_xgb_d6', p)


def export_mlp_87(kind):
    import t070
    torch.manual_seed(42); np.random.seed(42); torch.set_float32_matmul_precision('high')
    device = 'cuda'
    x = np.load(CACHE / 'x87.npy', mmap_mode='r')
    y = np.load(CACHE / 'y.npy')
    rng = np.random.default_rng(42)
    sample = x[np.sort(rng.choice(N, 32768, replace=False))]
    if kind == 't087_quantile':
        knots = fit_quantiles(sample)
        joblib.dump(knots, OUT / 'models' / 't087_quantile_knots.joblib')
        def enc(a): return q_transform(np.asarray(a), knots)
    else:
        q = np.nanpercentile(sample, [10, 50, 90], axis=0)
        loc = np.nan_to_num(q[1]).astype('f4')
        scale = np.nan_to_num((q[2] - q[0]) / 2.563).astype('f4')
        sd = np.nanstd(sample, axis=0)
        scale = np.where(scale > 1e-6, scale, np.where(sd > 1e-6, sd, 1.0)).astype('f4')
        np.savez_compressed(OUT / 'models' / 't087_gated_scaler.npz', loc=loc, scale=scale)
        def enc(a): return np.nan_to_num(np.clip((np.asarray(a) - loc) / scale, -8.0, 8.0), nan=0.0).astype('f4')
    del sample

    tr_t = torch.empty((N, 1141), device=device, dtype=torch.float16)
    for i in range(0, N, 4096):
        j = min(N, i + 4096)
        tr_t[i:j] = torch.from_numpy(enc(x[i:j])).to(device=device, dtype=torch.float16)
    tr_y = torch.from_numpy(y.astype('f4')).to(device)
    model = t070.Model(1141, 'gated').to(device)
    nn.init.constant_(model.head.bias, float(np.log(y.mean() / (1.0 - y.mean()))))
    opt = torch.optim.AdamW(model.parameters(), lr=4.5e-4, weight_decay=0.015)
    for ep in range(1, 9):
        model.train()
        for j in torch.randperm(N, device=device).split(2048):
            opt.zero_grad(set_to_none=True)
            F.binary_cross_entropy_with_logits(model(tr_t[j].float()), tr_y[j]).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
        if ep == 5:
            for g in opt.param_groups: g['lr'] = 1.5e-4
    torch.save({'state_dict': model.state_dict(), 'width': 1141, 'kind': 'gated'}, OUT / 'models' / f'{kind}.pt')
    del tr_t, tr_y; gc.collect(); torch.cuda.empty_cache()

    model.eval(); preds = []
    with torch.inference_mode():
        for i in range(N, N + V, 2048):
            j = min(N + V, i + 2048)
            preds.append(model(torch.from_numpy(enc(x[i:j])).to(device=device, dtype=torch.float16).float()).sigmoid().cpu().numpy())
    save_pred(kind, np.concatenate(preds))
    del model, opt; gc.collect(); torch.cuda.empty_cache()


def export_neural_96(data_dir, arm, seed):
    torch.manual_seed(seed); np.random.seed(seed); torch.set_float32_matmul_precision('high')
    device = 'cuda'
    xd = np.load(CACHE / 'dense96.npy', mmap_mode='r')
    xv = np.load(CACHE / 'vec96.npy', mmap_mode='r')
    y = np.load(CACHE / 'y.npy')

    # Build & save low-card categorical vocabularies once
    vocab_path = OUT / 'models' / 't096_cat_vocab.joblib'
    tr_f = pl.read_parquet(data_dir / 'train.parquet', columns=LOW_CATS)
    te_f = pl.read_parquet(data_dir / 'test.parquet', columns=LOW_CATS)
    tr_out, te_out, cards, vocab_dict = [], [], [], {}
    for col in LOW_CATS:
        s_tr, s_te = tr_f[col], te_f[col]
        vc = s_tr.drop_nulls().value_counts(name='cnt')
        if len(vc) > 1000:
            tr_out.append(np.zeros(N, dtype=np.int16)); te_out.append(np.zeros(V, dtype=np.int16)); cards.append(1); vocab_dict[col] = None; continue
        tab = vc.filter(pl.col('cnt') >= 10).sort(['cnt', col], descending=[True, False]).select(col).with_row_index('code', offset=1)
        tr_out.append(s_tr.to_frame().join(tab, on=col, how='left', maintain_order='left', validate='m:1')['code'].fill_null(0).to_numpy().astype(np.int16))
        te_out.append(s_te.to_frame().join(tab, on=col, how='left', maintain_order='left', validate='m:1')['code'].fill_null(0).to_numpy().astype(np.int16))
        cards.append(len(tab) + 1)
        vocab_dict[col] = dict(zip(tab[col].to_list(), tab['code'].to_list()))
    if not vocab_path.exists():
        joblib.dump({'vocab': vocab_dict, 'cards': cards}, vocab_path)
    tr_cat, te_cat = np.column_stack(tr_out), np.column_stack(te_out)
    del tr_f, te_f, tr_out, te_out

    rng = np.random.default_rng(42)
    samp = xd[np.sort(rng.choice(N, 65536, replace=False))]
    if arm == 't096_q_tabm':
        knots = fit_quantiles(samp.astype('f4'))
        if seed == 42:
            joblib.dump(knots, OUT / 'models' / 't096_q_tabm_knots.joblib')
        def enc(a): return q_transform(np.asarray(a), knots)
    else:
        s_d = samp.astype('d')
        med = np.nan_to_num(np.nanmedian(s_d, axis=0), nan=0.0)
        s_d = np.where(np.isfinite(s_d), s_d, med)
        sd = np.where(s_d.std(axis=0) > 1e-6, s_d.std(axis=0), 1.0)
        med = med.astype('f4'); sd = sd.astype('f4')
        np.savez_compressed(OUT / 'models' / 't096_dcnv2_r64_scaler.npz', med=med, sd=sd)
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
    if arm == 't096_dcnv2_r64':
        model = base94.DiverseNet(792, 14 * 34, cards, kind='dcnv2_r64', pos_rate=pos_rate).to(device)
        epochs, lr = 4, 4e-4
    else:
        model = base95.QuantileTabM(792, 14 * 34, cards, width=256, members=32, dropout=0.2, pos_rate=pos_rate).to(device)
        epochs, lr = 8, 4.5e-4

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.015)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=5e-5)
    for ep in range(1, epochs + 1):
        model.train()
        for j in torch.randperm(N, device=device).split(1024):
            opt.zero_grad(set_to_none=True)
            logits = model(tr_d[j].float(), tr_v[j].float(), tr_c[j])
            loss = F.binary_cross_entropy_with_logits(logits, tr_y[j].unsqueeze(1).expand_as(logits) if arm == 't096_q_tabm' else tr_y[j])
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
        sched.step()
    torch.save({'state_dict': model.state_dict(), 'cards': cards, 'arm': arm, 'seed': seed}, OUT / 'models' / f'{arm}_s{seed}.pt')
    del tr_d, tr_v, tr_c, tr_y; gc.collect(); torch.cuda.empty_cache()

    model.eval(); preds = []
    with torch.inference_mode():
        for i in range(0, V, 2048):
            j = min(V, i + 2048)
            d_t = torch.from_numpy(enc(xd[N + i:N + j])).to(device=device, dtype=torch.float16).float()
            v_t = torch.from_numpy(np.asarray(xv[N + i:N + j])).to(device=device, dtype=torch.float16).float()
            c_t = torch.from_numpy(te_cat[i:j]).to(device)
            out = model(d_t, v_t, c_t).sigmoid()
            if arm == 't096_q_tabm': out = out.mean(dim=1)
            preds.append(out.cpu().numpy())
    save_pred(f'{arm}_s{seed}', np.concatenate(preds))
    del model, opt; gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    (OUT / 'models').mkdir(exist_ok=True, parents=True)
    (OUT / 'preds').mkdir(exist_ok=True, parents=True)
    CACHE.mkdir(exist_ok=True, parents=True)
    data_dir = next(p.parent for p in Path('/kaggle/input').rglob('train.parquet') if (p.parent / 'sample_submission.csv').is_file())
    args = sys.argv[1:]; t0 = time.monotonic()
    if args == ['prepare']:
        prepare(data_dir)
    elif args[0] == 't096_xgb_d6':
        export_xgb_96()
    elif args[0] in ('t087_gated', 't087_quantile'):
        export_mlp_87(args[0])
    elif args[0] in ('t096_dcnv2_r64', 't096_q_tabm'):
        export_neural_96(data_dir, args[0], int(args[1]))
    else:
        export_xgb_87(args[0])
    print('EXPORTED', args, f'{time.monotonic()-t0:.0f}s', flush=True)
