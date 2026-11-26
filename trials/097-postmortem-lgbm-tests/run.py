"""T097: Post-competition local LightGBM forensic tests of row-only & cross-row ideas.

Runs on a 1/2 request-hash sample of train.parquet (full chronological cutoff preserved;
future-delta features computed on ALL 1,537,638 rows before subsampling so consecutive
item/user appearances are intact). Purely diagnostic, no submission.
"""
import gc, json, time
from pathlib import Path
import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]
TRIAL = Path(__file__).resolve().parent
CUTOFF = np.datetime64('2026-10-28T08:06:52.397')
VEC_COLS = [f'feature{i}' for i in [*range(26, 33), *range(40, 47)]]
HIST_DUR = [476, 477, 479, 480, 483, 484, 486, 487, 492, 493, 500, 501, 502, 503, 512, 513, 514, 515]
DRIFT_RATES = {'feature442', 'feature454', 'feature460', 'feature472'}
RAW_EPOCHS = {'feature57', 'feature58', 'feature59'}

PARAMS = dict(
    objective='binary', metric='auc', learning_rate=0.03, num_leaves=63,
    min_data_in_leaf=200, feature_fraction=0.45, bagging_fraction=0.8,
    bagging_freq=1, lambda_l1=2.0, lambda_l2=20.0, max_bin=127,
    num_threads=6, seed=42, verbosity=-1, force_col_wise=True,
)


def fit_eval(x_tr, y_tr, x_va, y_va, rounds=600, label=''):
    t0 = time.monotonic()
    dtr = lgb.Dataset(x_tr, label=y_tr, free_raw_data=True)
    dva = lgb.Dataset(x_va, label=y_va, reference=dtr, free_raw_data=True)
    m = lgb.train(
        PARAMS, dtr, num_boost_round=rounds,
        valid_sets=[dva], valid_names=['val'],
        callbacks=[lgb.early_stopping(60, verbose=False)],
    )
    p = m.predict(x_va, num_iteration=m.best_iteration, num_threads=6)
    auc = float(roc_auc_score(y_va, p))
    print(f'  [{label:26s}] AUC={auc:.6f}  best_iter={m.best_iteration:4d}  ({time.monotonic()-t0:.0f}s)', flush=True)
    return auc, m.best_iteration, p


def main():
    t_all = time.monotonic()
    raw = ROOT / 'data/train.parquet'
    schema = pl.scan_parquet(raw).collect_schema()
    num_cols = ['feature3'] + [
        c for c, t in schema.items()
        if c.startswith('feature') and c not in ('feature1', 'feature3')
        and str(t) in ('Float32', 'Int64', 'Int8')
    ]
    counter_cols = [c for c in num_cols if 154 <= int(c[7:]) <= 286]
    # Step 1: Compute FULL-DATASET future item/user counter deltas before subsampling
    print('Computing full-dataset future deltas on feature18 & feature5...', flush=True)
    delta_src_item = ['feature189', 'feature190', 'feature191', 'feature192', 'feature213', 'feature214', 'feature215', 'feature216', 'feature221', 'feature222', 'feature223', 'feature224']
    delta_src_user = ['feature60', 'feature66', 'feature67', 'feature68', 'feature69', 'feature70', 'feature73', 'feature79', 'feature80']
    meta_all = pl.read_parquet(raw, columns=['feature1', 'request_id', 'feature5', 'feature18', 'watched', *delta_src_item, *delta_src_user]).with_row_index('_row')

    # Item next-appearance deltas
    si = meta_all.sort(['feature18', 'feature1']).with_columns([
        (pl.col('feature1').shift(-1).over('feature18') - pl.col('feature1')).dt.total_seconds().alias('it_dt_next'),
        (pl.col('feature1') - pl.col('feature1').shift(1).over('feature18')).dt.total_seconds().alias('it_dt_prev'),
        *[(pl.col(c).shift(-1).over('feature18') - pl.col(c)).alias(f'it_dn_{c}') for c in delta_src_item],
    ]).with_columns([
        pl.when(pl.col('it_dt_next') > 0).then(pl.col(f'it_dn_{c}')).otherwise(None).alias(f'it_dn_{c}')
        for c in delta_src_item
    ]).with_columns([
        (pl.col('it_dn_feature213') / pl.col('it_dn_feature189').clip(lower_bound=1)).alias('it_drate_213'),
        (pl.col('it_dn_feature214') / pl.col('it_dn_feature190').clip(lower_bound=1)).alias('it_drate_214'),
        (pl.col('it_dn_feature224') / pl.col('it_dn_feature192').clip(lower_bound=1)).alias('it_drate_224'),
    ]).sort('_row')

    # User next-request deltas (computed at request level so intra-request rows don't mask next request)
    urq = meta_all.group_by(['feature5', 'request_id'], maintain_order=True).agg([
        pl.col('feature1').first(),
        *[pl.col(c).first() for c in delta_src_user],
    ]).sort(['feature5', 'feature1']).with_columns([
        (pl.col('feature1').shift(-1).over('feature5') - pl.col('feature1')).dt.total_seconds().alias('u_dt_next'),
        *[(pl.col(c).shift(-1).over('feature5') - pl.col(c)).alias(f'u_dn_{c}') for c in delta_src_user],
    ]).with_columns([
        pl.when(pl.col('u_dt_next') > 0).then(pl.col(f'u_dn_{c}')).otherwise(None).alias(f'u_dn_{c}')
        for c in delta_src_user
    ]).drop(['feature5', 'feature1', *delta_src_user])

    keep_mask = (meta_all['request_id'].hash(7) % 2 == 0).to_numpy()
    keep_rows = np.flatnonzero(keep_mask)
    future_item_cols = ['it_dt_next', 'it_dt_prev', *[f'it_dn_{c}' for c in delta_src_item], 'it_drate_213', 'it_drate_214', 'it_drate_224']
    future_item = si.filter(pl.Series(keep_mask)).select(future_item_cols).to_numpy().astype('f4')
    future_user = meta_all.filter(pl.Series(keep_mask)).select(['request_id']).join(urq, on='request_id', how='left', maintain_order='left').drop('request_id').to_numpy().astype('f4')
    del meta_all, si, urq
    gc.collect()

    # Step 2: Load the 1/2 sample with all numeric + vectors + request_id
    print('Loading 1/2 sample for row-only & cross-row tests...', flush=True)
    f = pl.scan_parquet(raw).filter(pl.col('request_id').hash(7) % 2 == 0).select(['feature1', 'request_id', 'feature5', 'feature18', 'watched', *num_cols, *VEC_COLS]).collect()
    assert len(f) == len(keep_rows)
    valid = (f['feature1'].to_numpy() >= CUTOFF)
    tr, va = np.flatnonzero(~valid), np.flatnonzero(valid)
    y = f['watched'].to_numpy().astype('f4')
    rq = f['request_id'].to_numpy()
    print(f'Sample rows: train={len(tr):,}, valid={len(va):,}', flush=True)

    # Build row-only base matrix: raw numeric + 49 cosines + 49 dot+bias + 36 duration + ages
    raw_num = np.column_stack([f[c].to_numpy().astype('f4') for c in num_cols])
    vecs = np.stack([f[c].fill_null([0.0]*32).list.to_array(32).to_numpy().astype('f4') for c in VEC_COLS], axis=1)
    norms = np.linalg.norm(vecs, axis=2)
    units = vecs / np.maximum(norms[:, :, None], 1e-6)
    bias = np.column_stack([f[f'feature{i}'].to_numpy().astype('f4') for i in [*range(33, 40), *range(47, 54)]])
    dots = (np.einsum('bik,bjk->bij', vecs[:, :7], vecs[:, 7:]) + bias[:, :7, None] + bias[:, None, 7:]).reshape(len(f), 49)
    cos49 = np.einsum('bik,bjk->bij', units[:, :7], units[:, 7:]).reshape(len(f), 49)
    d = np.where(f['feature54'].to_numpy().astype('d') > 0, f['feature54'].to_numpy().astype('d'), np.nan)
    s = np.column_stack([f[f'feature{i}'].to_numpy().astype('d') for i in HIST_DUR])
    s = np.where(s >= 0, s, np.nan)
    dur = np.column_stack([np.log1p(s) - np.log1p(d[:, None]), s - d[:, None]]).astype('f4')
    now_s = f['feature1'].dt.epoch('ms').to_numpy().astype('d') / 1000.0
    age58_h = np.maximum((now_s - f['feature58'].to_numpy().astype('d')) / 3600.0, 0.0)
    age59_h = np.maximum((now_s - f['feature59'].to_numpy().astype('d') / 1000.0) / 3600.0, 0.0)
    age57_d = np.maximum((now_s - np.where(f['feature57'].to_numpy().astype('d') > 1e11, f['feature57'].to_numpy().astype('d') / 1000.0, f['feature57'].to_numpy().astype('d'))) / 86400.0, 0.0)
    ages = np.column_stack([np.log1p(age58_h), np.log1p(age59_h), np.log1p(age57_d)]).astype('f4')
    del vecs, units
    gc.collect()

    X_row = np.ascontiguousarray(np.column_stack([raw_num, cos49, dots, norms, dur, ages]), dtype=np.float32)

    # === TEST 1: ROW-ONLY IDEAS ===
    print('\n=== TEST 1: ROW-ONLY IDEAS ===', flush=True)
    res = {}
    res['row_control'], _, _ = fit_eval(X_row[tr], y[tr], X_row[va], y[va], label='1A. Row-only control')

    # 1B. Drift-controlled (SumanjaliSegu): replace counters 154..286 with log1p(c)-log1p(age58_h), drop raw epochs & drifting rates
    drop_idx = {num_cols.index(c) for c in (DRIFT_RATES | RAW_EPOCHS)}
    cnt_idx = [num_cols.index(c) for c in counter_cols]
    keep_raw = [i for i in range(len(num_cols)) if i not in drop_idx and i not in set(cnt_idx)]
    log_age = np.log1p(age58_h)[:, None]
    norm_cnt = (np.log1p(np.maximum(raw_num[:, cnt_idx].astype('d'), 0.0)) - log_age).astype('f4')
    X_drift = np.ascontiguousarray(np.column_stack([raw_num[:, keep_raw], norm_cnt, cos49, dots, norms, dur, ages]), dtype=np.float32)
    res['row_drift_ctrl'], _, _ = fit_eval(X_drift[tr], y[tr], X_drift[va], y[va], label='1B. Drift-controlled')
    del X_drift, norm_cnt; gc.collect()

    # Build within-request context features for Test 1C (teacher) and Test 2
    rq_cols = ['feature54', 'feature418', 'feature419', 'feature411', 'feature410', 'feature430', 'feature431', 'feature417', 'feature352', 'feature346', 'feature480', 'feature493', 'feature515', 'feature53', 'feature50']
    rq_df = f.select(['request_id', *rq_cols]).with_columns([
        pl.Series('cos_32_43', cos49[:, 6*7 + 3]),
        pl.Series('cos_30_46', cos49[:, 4*7 + 6]),
        pl.Series('dot_32_43', dots[:, 6*7 + 3]),
        pl.Series('dot_30_46', dots[:, 4*7 + 6]),
    ])
    rank_targets = rq_cols + ['cos_32_43', 'cos_30_46', 'dot_32_43', 'dot_30_46']
    size_expr = pl.len().over('request_id')
    exprs = [size_expr.alias('rq_size')]
    for c in rank_targets:
        exprs += [
            (pl.col(c).rank('average').over('request_id') / size_expr).alias(f'rq_rk_{c}'),
            (pl.col(c) - pl.col(c).mean().over('request_id')).alias(f'rq_dv_{c}'),
            (pl.col(c) - pl.col(c).max().over('request_id')).alias(f'rq_gp_{c}'),
        ]
    rq_feats = rq_df.select(exprs).to_numpy().astype('f4')

    # Extra 2nd-best margin + position_in_request for Test 2B
    pos_in_rq = f.select(pl.col('request_id').cum_count().over('request_id').cast(pl.Float32)).to_numpy()
    # 2nd-best margin on top 6 signals
    top6 = ['feature54', 'feature418', 'feature419', 'cos_32_43', 'cos_30_46', 'dot_32_43']
    m2_list = [pos_in_rq, pos_in_rq / np.maximum(rq_feats[:, 0:1], 1.0)]
    for c in top6:
        arr = rq_df[c].to_numpy().astype('d')
        # Compute max and 2nd-max per request
        rq_codes = np.unique(rq, return_inverse=True)[1]
        order = np.lexsort((-np.nan_to_num(arr, nan=-1e9), rq_codes))
        starts = np.r_[0, np.flatnonzero(np.diff(rq_codes[order])) + 1]
        best_i = order[starts]
        sec_i = order[np.minimum(starts + 1, len(arr) - 1)]
        other_best = np.where(np.arange(len(arr)) == best_i[rq_codes], arr[sec_i[rq_codes]], arr[best_i[rq_codes]])
        other_best = np.where(rq_feats[:, 0] > 1, other_best, np.nan)
        m2_list.append((arr - other_best).astype('f4')[:, None])
    X_slate_extra = np.column_stack(m2_list).astype('f4')

    # 1C. Training-only distillation (SumanjaliSegu): 4-fold OOF request-aware teacher -> row-only student!
    print('Training 4-fold OOF request-aware teacher on train split only...', flush=True)
    X_rq = np.ascontiguousarray(np.column_stack([X_row, rq_feats]), dtype=np.float32)
    oof_teacher = np.zeros(len(tr), dtype=np.float32)
    for k, (a, b) in enumerate(GroupKFold(4).split(tr, y[tr], groups=rq[tr])):
        dtr = lgb.Dataset(X_rq[tr[a]], label=y[tr[a]], free_raw_data=True)
        mt = lgb.train(PARAMS, dtr, num_boost_round=350)
        oof_teacher[b] = mt.predict(X_rq[tr[b]], num_threads=6)
    print(f'  Teacher 4-fold OOF AUC on train: {roc_auc_score(y[tr], oof_teacher):.6f}', flush=True)
    y_soft = 0.5 * y[tr] + 0.5 * oof_teacher
    res['row_distilled'], _, _ = fit_eval(X_row[tr], y_soft, X_row[va], y[va], label='1C. Row-only distilled')

    # === TEST 2: CROSS-ROW / FUTURE-DELTA FORENSICS ===
    print('\n=== TEST 2: CROSS-ROW & FUTURE-DELTA FORENSICS ===', flush=True)
    res['rq_control'], _, _ = fit_eval(X_rq[tr], y[tr], X_rq[va], y[va], label='2A. Request-ctx control')

    X_rq_slate = np.ascontiguousarray(np.column_stack([X_rq, X_slate_extra]), dtype=np.float32)
    res['rq_plus_margin2_pos'], _, _ = fit_eval(X_rq_slate[tr], y[tr], X_rq_slate[va], y[va], label='2B. +pos & 2nd-best gap')
    del X_rq_slate; gc.collect()

    X_rq_future = np.ascontiguousarray(np.column_stack([X_rq, future_item, future_user]), dtype=np.float32)
    res['rq_plus_future_deltas'], _, _ = fit_eval(X_rq_future[tr], y[tr], X_rq_future[va], y[va], label='2C. +future item/user d')

    res['seconds'] = time.monotonic() - t_all
    (TRIAL / 'results.json').write_text(json.dumps(res, indent=2) + '\n')
    print('\nSummary:', json.dumps(res, indent=2), flush=True)


if __name__ == '__main__':
    main()
