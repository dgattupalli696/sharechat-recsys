"""T093: Shallow long-horizon XGBoost (d4 stationary, d5 long) + global rank:pairwise on 1,240 super-enriched row-only features."""
import gc, json, sys, time
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
import t092_worker as base92

OUT = Path('/kaggle/working/t093')
CACHE = base92.CACHE


def fit_arm(fold, arm):
    import cupy as cp
    import xgboost as xgb
    tr, va, y = base92.get_split(fold)
    x = np.load(CACHE/'x.npy', mmap_mode='r')
    # Identify raw timestamp columns (feature57, feature58, feature59) inside the first 584 numeric columns
    raw_path = next(p for p in Path('/kaggle/input').rglob('train.parquet') if (p.parent/'sample_submission.csv').is_file())
    num_cols = base92.numeric_cols(pl.scan_parquet(raw_path).collect_schema())
    drop_idx = {num_cols.index(c) for c in ('feature57', 'feature58', 'feature59')}
    keep_cols = np.array([i for i in range(x.shape[1]) if i not in drop_idx], dtype=np.int64) if arm == 'xgb_d4_stat' else None

    def get_x(j):
        return x[j][:, keep_cols] if keep_cols is not None else x[j]

    is_rank = (arm == 'xgb_d5_rank')
    if arm == 'xgb_d4_stat':
        params = dict(objective='binary:logistic', tree_method='hist', device='cuda', max_depth=4, eta=0.015, subsample=0.8, colsample_bytree=0.25, colsample_bynode=0.5, min_child_weight=250, reg_lambda=50, reg_alpha=8, gamma=1, max_bin=127, seed=42, nthread=4)
        rounds_list = (2400, 3200, 4200)
    elif arm == 'xgb_d5_long':
        params = dict(objective='binary:logistic', tree_method='hist', device='cuda', max_depth=5, eta=0.015, subsample=0.8, colsample_bytree=0.30, colsample_bynode=0.5, min_child_weight=200, reg_lambda=40, reg_alpha=6, gamma=1, max_bin=127, seed=45, nthread=4)
        rounds_list = (2000, 2800, 3600)
    elif arm == 'xgb_d5_rank':
        params = dict(objective='rank:pairwise', lambdarank_pair_method='mean', lambdarank_num_pair_per_sample=8, tree_method='hist', device='cuda', max_depth=5, eta=0.02, subsample=0.8, colsample_bytree=0.35, colsample_bynode=0.5, min_child_weight=200, reg_lambda=30, reg_alpha=5, gamma=1, max_bin=127, seed=46, nthread=4)
        rounds_list = (1200, 1800, 2400)
    else:
        raise ValueError(arm)

    class Iter(xgb.DataIter):
        def __init__(self): self.reset(); super().__init__()
        def reset(self): self.i = 0
        def next(self, input_data):
            if self.i >= len(tr): return False
            j = tr[self.i:self.i+8192]; self.i += 8192
            kw = dict(data=cp.asarray(get_x(j)), label=cp.asarray(y[j]))
            if is_rank:
                kw['qid'] = cp.zeros(len(j), dtype=cp.uint32)
            input_data(**kw)
            return True

    dm = xgb.QuantileDMatrix(Iter(), max_bin=127, nthread=4)
    model = xgb.train(params, dm, max(rounds_list))
    del dm; gc.collect(); cp.get_default_memory_pool().free_all_blocks()
    res = {}
    for r in rounds_list:
        def pred(j):
            out = cp.asnumpy(model.inplace_predict(cp.asarray(get_x(j)), iteration_range=(0, r)))
            return (1.0 / (1.0 + np.exp(-np.clip(out, -20, 20)))) if is_rank else out
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
        base92.prepare(raw); res = {}
    else:
        res = fit_arm(args[0], args[1])
    res['seconds'] = time.monotonic() - t0
    (OUT/('-'.join(args)+'.json')).write_text(json.dumps(res) + '\n')
