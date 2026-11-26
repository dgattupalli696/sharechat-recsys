"""T092: Super-enriched row-only XGBoost (d5/d6/d7) + CatBoost GPU (d6/d8 with low-card cats)."""
import gc, json, sys, time
from pathlib import Path
import numpy as np
import polars as pl
import pyarrow.parquet as pq
from scipy.special import logit
from sklearn.metrics import roc_auc_score

OUT = Path('/kaggle/working/t092')
CACHE = Path('/kaggle/temp/t092')
PRIMARY_CUT = np.datetime64('2026-10-28T08:06:52.397')
EARLIER_CUT = np.datetime64('2026-10-27T08:06:52.397')
EARLIER_END = np.datetime64('2026-10-27T18:00:00')
LOW_CATS = [f'feature{i}' for i in [6,7,8,9,10,11,12,13,14,16,17,19,21,22,25]]
HIST_DUR = [476,477,479,480,483,484,486,487,492,493,500,501,502,503,512,513,514,515]
VEC_COLS = [f'feature{i}' for i in [*range(26,33), *range(40,47)]]
USER_RATES = ['feature352', 'feature346', 'feature358', 'feature334']
ITEM_RATES = ['feature418', 'feature419', 'feature430', 'feature411']


def numeric_cols(schema):
    return ['feature3'] + [
        c for c, t in schema.items()
        if c.startswith('feature') and c not in ('feature1', 'feature3')
        and str(t) in ('Float32', 'Int64', 'Int8')
    ]


def safe_logit(a):
    return logit(np.clip(np.where(np.isfinite(a), a, np.nan), 1e-5, 1.0 - 1e-5))


def super_features(f, num_cols):
    n = len(f)
    raw = np.column_stack([f[c].to_numpy().astype('f4') for c in num_cols])
    vecs = np.stack([f[c].fill_null([0.0]*32).list.to_array(32).to_numpy().astype('f4') for c in VEC_COLS], axis=1)
    norms = np.linalg.norm(vecs, axis=2)
    units = vecs / np.maximum(norms[:, :, None], 1e-6)
    bias = np.column_stack([f[f'feature{i}'].to_numpy().astype('f4') for i in [*range(33,40), *range(47,54)]])
    dots = np.einsum('bik,bjk->bij', vecs[:, :7], vecs[:, 7:]) + bias[:, :7, None] + bias[:, None, 7:]
    cos49 = np.einsum('bik,bjk->bij', units[:, :7], units[:, 7:])
    # 36 duration interactions
    d = f['feature54'].to_numpy().astype('d')
    d = np.where(d > 0, d, np.nan)
    s = np.column_stack([f[f'feature{i}'].to_numpy().astype('d') for i in HIST_DUR])
    s = np.where(s >= 0, s, np.nan)
    dur = np.column_stack([np.log1p(s) - np.log1p(d[:, None]), s - d[:, None]]).astype('f4')
    # Stationary ages + time-of-day
    now = f['feature1'].dt.epoch('ms').to_numpy().astype('d') / 1000.0
    st57 = f['feature57'].to_numpy().astype('d')
    st57 = np.where(st57 > 1e11, st57 / 1000.0, st57)
    st58 = f['feature58'].to_numpy().astype('d')
    st59 = f['feature59'].to_numpy().astype('d') / 1000.0
    ages = [np.log1p(np.where((st > 0) & (now >= st), (now - st) / 86400.0, np.nan)) for st in (st57, st58, st59)]
    hr = (now % 86400.0) / 86400.0
    # User x Item watch-rate interactions (32 features: product + logit sum)
    ur = [f[c].to_numpy().astype('d') for c in USER_RATES]
    ir = [f[c].to_numpy().astype('d') for c in ITEM_RATES]
    rate_cross = []
    for u in ur:
        lu = safe_logit(u)
        for v in ir:
            rate_cross.append(u * v)
            rate_cross.append(lu + safe_logit(v))
    # Top cosine x rate/duration interactions (11 features)
    c32_43 = cos49[:, 6, 3].astype('d')
    c30_46 = cos49[:, 4, 6].astype('d')
    l418 = safe_logit(ir[0])
    l352 = safe_logit(ur[0])
    ld = np.log1p(d)
    l493 = np.log1p(np.maximum(f['feature493'].to_numpy().astype('d'), 0))
    cos_cross = [
        c32_43 * l418, c32_43 * l352, c32_43 * ld, c32_43 * l493,
        c30_46 * l418, c30_46 * l352, c30_46 * ld, c30_46 * l493,
        c32_43 + c30_46, c32_43 - c30_46, c32_43 * c30_46,
    ]
    # User watch-time vs item watch-time log ratios (9 features)
    uwt = [np.log1p(np.maximum(f[f'feature{k}'].to_numpy().astype('d'), 0)) for k in (493, 480, 484)]
    iwt = [np.log1p(np.maximum(f[f'feature{k}'].to_numpy().astype('d'), 0)) for k in (515, 511, 502)]
    wt_cross = [u - v for u in uwt for v in iwt]
    # Watch-to-skip ratios (3 features)
    ratios = [
        ur[0] / (np.maximum(ur[2], 0) + 1e-4),
        ur[1] / (np.maximum(ur[3], 0) + 1e-4),
        ir[0] / (np.maximum(ir[2], 0) + 1e-4),
    ]
    extra = np.column_stack([*ages, np.sin(2*np.pi*hr), np.cos(2*np.pi*hr), *rate_cross, *cos_cross, *wt_cross, *ratios]).astype('f4')
    return np.column_stack([raw, vecs.reshape(n, 448), norms, dots.reshape(n, 49), cos49.reshape(n, 49), dur, extra]).astype('f4')


def prepare(raw_path):
    schema = pl.scan_parquet(raw_path).collect_schema()
    num_cols = numeric_cols(schema)
    assert 'request_id' not in num_cols and len(num_cols) == 584
    fields = ['feature1'] + num_cols + VEC_COLS
    pf = pq.ParquetFile(raw_path)
    n = pf.metadata.num_rows
    assert n == 1537638
    width = 584 + 448 + 14 + 49 + 49 + 36 + 60
    x = np.lib.format.open_memmap(CACHE/'x.npy', mode='w+', dtype='f4', shape=(n, width))
    off = 0
    for batch in pf.iter_batches(columns=fields, batch_size=8192):
        f = pl.from_arrow(batch)
        z = super_features(f, num_cols)
        if off == 0:
            z1 = super_features(f.slice(0, 1), num_cols)
            np.testing.assert_allclose(z[:1], z1, atol=1e-5, equal_nan=True)
        x[off:off+len(f)] = z
        off += len(f)
    assert off == n
    x.flush()
    meta = pl.read_parquet(raw_path, columns=['feature1', 'watched'])
    np.save(CACHE/'t.npy', meta['feature1'].to_numpy())
    np.save(CACHE/'y.npy', meta['watched'].to_numpy())
    print(f'Prepared super matrix {x.shape}', flush=True)


def get_split(fold):
    t = np.load(CACHE/'t.npy'); y = np.load(CACHE/'y.npy')
    cut = PRIMARY_CUT if fold == 'primary' else EARLIER_CUT
    tr = np.flatnonzero(t < cut)
    va = np.flatnonzero((t >= cut) & ((fold == 'primary') | (t < EARLIER_END)))
    assert (len(tr), len(va)) == ((1230110, 307528) if fold == 'primary' else (710330, 299969))
    return tr, va, y


def fit_xgb(fold, cfg):
    import cupy as cp
    import xgboost as xgb
    tr, va, y = get_split(fold)
    x = np.load(CACHE/'x.npy', mmap_mode='r')
    specs = {
        'xgb_d6': (dict(max_depth=6, eta=0.02, colsample_bytree=0.4, colsample_bynode=0.5, min_child_weight=150, reg_lambda=20, reg_alpha=2, gamma=1, seed=42), (900, 1200, 1500)),
        'xgb_d5': (dict(max_depth=5, eta=0.02, colsample_bytree=0.35, colsample_bynode=0.5, min_child_weight=200, reg_lambda=30, reg_alpha=5, gamma=1, seed=43), (1200, 1600, 2000)),
        'xgb_d7': (dict(max_depth=7, eta=0.02, colsample_bytree=0.4, colsample_bynode=0.4, min_child_weight=150, reg_lambda=25, reg_alpha=3, gamma=1, seed=44), (900, 1200)),
    }
    p_cfg, rounds_list = specs[cfg]
    params = dict(objective='binary:logistic', tree_method='hist', device='cuda', subsample=0.8, max_bin=127, nthread=4, **p_cfg)

    class Iter(xgb.DataIter):
        def __init__(self): self.reset(); super().__init__()
        def reset(self): self.i = 0
        def next(self, input_data):
            if self.i >= len(tr): return False
            j = tr[self.i:self.i+8192]; self.i += 8192
            input_data(data=cp.asarray(x[j]), label=cp.asarray(y[j]))
            return True

    dm = xgb.QuantileDMatrix(Iter(), max_bin=127, nthread=4)
    model = xgb.train(params, dm, max(rounds_list))
    del dm; gc.collect(); cp.get_default_memory_pool().free_all_blocks()
    res = {}
    for r in rounds_list:
        def pred(j): return cp.asnumpy(model.inplace_predict(cp.asarray(x[j]), iteration_range=(0, r)))
        p = np.concatenate([pred(va[i:i+8192]) for i in range(0, len(va), 8192)]).astype('f4')
        np.testing.assert_allclose(p[:8], np.concatenate([pred(j[None]) for j in va[:8]]), atol=1e-6)
        auc = float(roc_auc_score(y[va], p))
        res[str(r)] = auc
        print(f'{fold} {cfg} r={r} auc={auc:.6f}', flush=True)
        np.savez_compressed(OUT/f'{fold}-{cfg}-r{r}.npz', row_ids=va, y_true=y[va], probabilities=p)
    return res


def fit_cat(raw_path, fold, cfg):
    from catboost import CatBoostClassifier, FeaturesData, Pool
    tr, va, y = get_split(fold)
    x = np.load(CACHE/'x.npy', mmap_mode='r')
    # Only low-cardinality categoricals (<= 1000 unique in train, strictly excluding user/item/creator IDs)
    df_cat = pl.read_parquet(raw_path, columns=LOW_CATS)
    keep_cats = []
    cat_cols_arr = []
    for col in LOW_CATS:
        s = df_cat[col]
        if s.gather(tr).drop_nulls().n_unique() <= 1000:
            keep_cats.append(col)
            cat_cols_arr.append(s.fill_null('__NA__').to_numpy().astype(str))
    cat_mat = np.column_stack(cat_cols_arr).astype(object)
    del df_cat, cat_cols_arr; gc.collect()

    depth, iters, l2, seed = (6, 3000, 15.0, 42) if cfg == 'cat_d6' else (8, 2200, 25.0, 43)
    tr_pool = Pool(FeaturesData(num_feature_data=np.ascontiguousarray(x[tr], dtype=np.float32), cat_feature_data=cat_mat[tr]), label=y[tr], thread_count=4)
    model = CatBoostClassifier(
        iterations=iters, learning_rate=0.03, depth=depth, l2_leaf_reg=l2,
        border_count=128, random_seed=seed, task_type='GPU', devices='0',
        gpu_ram_part=0.85, max_ctr_complexity=2, verbose=500, allow_writing_files=False,
    )
    model.fit(tr_pool)
    del tr_pool; gc.collect()

    va_pool = Pool(FeaturesData(num_feature_data=np.ascontiguousarray(x[va], dtype=np.float32), cat_feature_data=cat_mat[va]), thread_count=4)
    p = model.predict_proba(va_pool)[:, 1].astype('f4')
    p1 = model.predict_proba(Pool(FeaturesData(num_feature_data=np.ascontiguousarray(x[va[:1]], dtype=np.float32), cat_feature_data=cat_mat[va[:1]]), thread_count=4))[:, 1].astype('f4')
    np.testing.assert_allclose(p[:1], p1, atol=1e-6)
    auc = float(roc_auc_score(y[va], p))
    print(f'{fold} {cfg} iters={iters} cats={len(keep_cats)} auc={auc:.6f}', flush=True)
    np.savez_compressed(OUT/f'{fold}-{cfg}.npz', row_ids=va, y_true=y[va], probabilities=p)
    del va_pool, model, cat_mat; gc.collect()
    return {'auc': auc, 'cats': keep_cats}


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True, parents=True); CACHE.mkdir(exist_ok=True, parents=True)
    raw = next(p for p in Path('/kaggle/input').rglob('train.parquet') if (p.parent/'sample_submission.csv').is_file())
    args = sys.argv[1:]; t0 = time.monotonic()
    if args == ['prepare']:
        prepare(raw); res = {}
    elif args[1].startswith('xgb'):
        res = fit_xgb(args[0], args[1])
    else:
        res = fit_cat(raw, args[0], args[1])
    res['seconds'] = time.monotonic() - t0
    (OUT/('-'.join(args)+'.json')).write_text(json.dumps(res) + '\n')
