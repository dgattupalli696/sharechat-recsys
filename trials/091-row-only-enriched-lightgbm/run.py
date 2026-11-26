"""T091: Local CPU enriched row-only LightGBM (standard leaves-63 + extra_trees leaves-127)."""
import gc, json, os, runpy, sys, time
from pathlib import Path
import lightgbm as lgb
import numpy as np
import polars as pl
import pyarrow.parquet as pq
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

T = Path(__file__).resolve().parent
ROOT = T.parents[1]
OUT = ROOT / 'artifacts/local-t091'
sys.path.insert(0, str(ROOT))
t090 = runpy.run_path(str(T.parent / '090-enriched-row-only-tabm/worker.py'))

PRIMARY_CUT = np.datetime64('2026-10-28T08:06:52.397')
EARLIER_CUT = np.datetime64('2026-10-27T08:06:52.397')
EARLIER_END = np.datetime64('2026-10-27T18:00:00')
LOW_CATS = t090['LOW_CATS']
VEC_COLS = t090['VEC_COLS']


def prepare_cache():
    raw = ROOT / 'data/train.parquet'
    schema = pl.scan_parquet(raw).collect_schema()
    num_cols = t090['numeric_cols'](schema)
    fields = ['feature1'] + num_cols + VEC_COLS
    pf = pq.ParquetFile(raw)
    n = pf.metadata.num_rows
    assert n == 1537638
    n_dense = 584 + 49 + 49 + 36 + 11
    xd_path = OUT / 'dense.npy'
    if not xd_path.exists():
        xd = np.lib.format.open_memmap(xd_path, mode='w+', dtype='f4', shape=(n, n_dense))
        off = 0
        for batch in pf.iter_batches(columns=fields, batch_size=8192):
            f = pl.from_arrow(batch)
            d, _ = t090['row_features'](f, num_cols)
            xd[off:off+len(f)] = d
            off += len(f)
        assert off == n
        xd.flush()
        del xd
    meta = pl.read_parquet(raw, columns=['feature1', 'watched'])
    return meta['feature1'].to_numpy(), meta['watched'].to_numpy(), n_dense


def fit_fold(fold, t_arr, y_arr, n_dense):
    raw = ROOT / 'data/train.parquet'
    cut = PRIMARY_CUT if fold == 'primary' else EARLIER_CUT
    tr = np.flatnonzero(t_arr < cut)
    va = np.flatnonzero((t_arr >= cut) & ((fold == 'primary') | (t_arr < EARLIER_END)))
    assert (len(tr), len(va)) == ((1230110, 307528) if fold == 'primary' else (710330, 299969))
    cats, _ = t090['build_cats'](raw, tr)
    xd = np.load(OUT / 'dense.npy', mmap_mode='r')
    cat_idx = list(range(n_dense, n_dense + cats.shape[1]))

    x_tr = np.empty((len(tr), n_dense + cats.shape[1]), dtype=np.float32)
    x_tr[:, :n_dense] = xd[tr]
    x_tr[:, n_dense:] = cats[tr]
    x_va = np.empty((len(va), n_dense + cats.shape[1]), dtype=np.float32)
    x_va[:, :n_dense] = xd[va]
    x_va[:, n_dense:] = cats[va]
    del cats
    gc.collect()

    configs = {
        'lgb_l63': (
            dict(
                objective='binary', metric='auc', learning_rate=0.025, num_leaves=63,
                min_data_in_leaf=200, feature_fraction=0.35, bagging_fraction=0.8,
                bagging_freq=1, lambda_l1=5.0, lambda_l2=20.0, max_bin=127,
                cat_smooth=30.0, max_cat_to_onehot=8, num_threads=6, seed=42,
                verbosity=-1, force_col_wise=True,
            ),
            (1000, 1400, 1800),
        ),
        'lgb_l127_et': (
            dict(
                objective='binary', metric='auc', learning_rate=0.025, num_leaves=127,
                extra_trees=True, min_data_in_leaf=200, feature_fraction=0.40,
                bagging_fraction=0.8, bagging_freq=1, lambda_l1=5.0, lambda_l2=30.0,
                max_bin=127, cat_smooth=30.0, max_cat_to_onehot=8, num_threads=6,
                seed=43, verbosity=-1, force_col_wise=True,
            ),
            (1200, 1600, 2000),
        ),
    }

    for name, (params, rounds_list) in configs.items():
        began = time.monotonic()
        dtrain = lgb.Dataset(x_tr, label=y_arr[tr], categorical_feature=cat_idx, free_raw_data=False)
        model = lgb.train(params, dtrain, num_boost_round=max(rounds_list))
        for r in rounds_list:
            p = model.predict(x_va, num_iteration=r, num_threads=6).astype(np.float32)
            p_one = model.predict(x_va[:1], num_iteration=r, num_threads=6).astype(np.float32)
            np.testing.assert_allclose(p[:1], p_one, atol=1e-6, rtol=1e-6)
            auc = float(roc_auc_score(y_arr[va], p))
            print(f'{fold} {name} r={r} auc={auc:.6f} ({time.monotonic()-began:.0f}s)', flush=True)
            np.savez_compressed(OUT / f'{fold}-{name}-r{r}.npz', row_ids=va, y_true=y_arr[va], probabilities=p)
        del dtrain, model
        gc.collect()
    del x_tr, x_va
    gc.collect()


def score_all():
    m = runpy.run_path(str(T.parent / '065-row-only-blend-weights/run.py'))
    blend_s024 = runpy.run_path(str(T.parent / '087-full-row-only-landmark-blend/assemble.py'))['blend']
    art = ROOT / 'artifacts'
    report = {}
    for fold in ['earlier', 'primary']:
        ids, y, anchor, p = m['fold'](fold)
        def get(tr, k):
            i, l, v = m['read'](art / f'kaggle-t{tr}/t{tr}/{fold}-{k}.npz')
            np.testing.assert_array_equal(ids, i)
            return v
        q = np.column_stack([
            anchor, *[p[k] for k in m['trees']],
            get('069', '1-2400'), get('070', 'gated-8'), get('072', 'xgb-base'),
        ])
        s024 = blend_s024(q, get('081', 'quantile-8'), get('083', 'paired'), get('086', 'learned'), get('086', 'random'))
        s024_auc = float(roc_auc_score(y, s024))
        preds, standalone = {}, {}
        for path in sorted(OUT.glob(f'{fold}-lgb_*.npz')):
            key = path.stem[len(fold) + 1:]
            i, l, v = m['read'](path)
            np.testing.assert_array_equal(ids, i)
            preds[key] = v
            standalone[key] = float(roc_auc_score(y, v))
        if 'lgb_l63-r1400' in preds and 'lgb_l127_et-r1600' in preds:
            preds['lgb_mean_mid'] = 0.5 * (preds['lgb_l63-r1400'] + preds['lgb_l127_et-r1600'])
            standalone['lgb_mean_mid'] = float(roc_auc_score(y, preds['lgb_mean_mid']))
        if 'lgb_l63-r1800' in preds and 'lgb_l127_et-r2000' in preds:
            preds['lgb_mean_long'] = 0.5 * (preds['lgb_l63-r1800'] + preds['lgb_l127_et-r2000'])
            standalone['lgb_mean_long'] = float(roc_auc_score(y, preds['lgb_mean_long']))
        grid = {}
        z_base = logit(np.clip(s024, 1e-8, 1 - 1e-8))
        for k, v in preds.items():
            z_v = logit(np.clip(v, 1e-8, 1 - 1e-8))
            for wt in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35):
                grid[f'{k}/prob/{wt}'] = float(roc_auc_score(y, (1 - wt) * s024 + wt * v))
                grid[f'{k}/logit/{wt}'] = float(roc_auc_score(y, expit((1 - wt) * z_base + wt * z_v)))
        if fold == 'earlier':
            winner = max(grid, key=grid.get)
        report[fold] = {
            's024_auc': s024_auc,
            'standalone': standalone,
            'selected': winner,
            'blend_auc': grid[winner],
            'delta_vs_s024': grid[winner] - s024_auc,
            'top5': sorted(((v - s024_auc, k, v) for k, v in grid.items()), reverse=True)[:5],
        }
    (T / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    for fold in ['earlier', 'primary']:
        r = report[fold]
        print(f"SCORE {fold}: S024={r['s024_auc']:.6f} -> {r['selected']} = {r['blend_auc']:.6f} (delta={r['delta_vs_s024']:+.6f})", flush=True)


def main():
    os.chdir(ROOT)
    OUT.mkdir(exist_ok=True, parents=True)
    t_arr, y_arr, n_dense = prepare_cache()
    for fold in ['earlier', 'primary']:
        fit_fold(fold, t_arr, y_arr, n_dense)
    score_all()
    # Clean up 4.5GB dense cache after scoring to preserve local disk space
    (OUT / 'dense.npy').unlink(missing_ok=True)


if __name__ == '__main__':
    main()
