"""Score T095 quantile-PLR TabM & DCNv2-MoE4 + joint ensemble across S024 + T090..T095."""
import json, runpy, sys
from pathlib import Path
import numpy as np
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

t = Path(__file__).resolve().parent
root = t.parents[1]
art = root / 'artifacts'
m = runpy.run_path(str(t.parent / '065-row-only-blend-weights/run.py'))
blend_s024 = runpy.run_path(str(t.parent / '087-full-row-only-landmark-blend/assemble.py'))['blend']
folder = Path(sys.argv[1])

def load_pool(fold):
    ids, y, anchor, p = m['fold'](fold)
    def get(path):
        i, l, v = m['read'](path)
        np.testing.assert_array_equal(ids, i)
        np.testing.assert_array_equal(y, l)
        return v
    q = np.column_stack([
        anchor, *[p[k] for k in m['trees']],
        get(art / f'kaggle-t069/t069/{fold}-1-2400.npz'),
        get(art / f'kaggle-t070/t070/{fold}-gated-8.npz'),
        get(art / f'kaggle-t072/t072/{fold}-xgb-base.npz'),
    ])
    s024 = blend_s024(
        q,
        get(art / f'kaggle-t081/t081/{fold}-quantile-8.npz'),
        get(art / f'kaggle-t083/t083/{fold}-paired.npz'),
        get(art / f'kaggle-t086/t086/{fold}-learned.npz'),
        get(art / f'kaggle-t086/t086/{fold}-random.npz'),
    )
    pool = {
        's024': s024,
        't090_dcn4': get(art / f'kaggle-t090/t090/{fold}-dcnv2-w128-s42-ep4.npz'),
        't092_d5_2000': get(art / f'kaggle-t092/t092/{fold}-xgb_d5-r2000.npz'),
        't092_d6_900': get(art / f'kaggle-t092/t092/{fold}-xgb_d6-r900.npz'),
        't092_d6_1200': get(art / f'kaggle-t092/t092/{fold}-xgb_d6-r1200.npz'),
        't094_r64_ep4': get(art / f'kaggle-t094/t094/{fold}-dcnv2_r64-ep4.npz'),
        't094_moe4_ep3': get(art / f'kaggle-t094/t094/{fold}-dcnv2_moe4-ep3.npz'),
        't094_moe4_ep4': get(art / f'kaggle-t094/t094/{fold}-dcnv2_moe4-ep4.npz'),
        't094_rank5_r500': get(art / f'kaggle-t094/t094/{fold}-rank_d5-r500.npz'),
        't094_rank5_r700': get(art / f'kaggle-t094/t094/{fold}-rank_d5-r700.npz'),
        't094_rank5_r900': get(art / f'kaggle-t094/t094/{fold}-rank_d5-r900.npz'),
    }
    st = {}
    for path in sorted(folder.glob(f'{fold}-*.npz')):
        key = 't095_' + path.stem[len(fold) + 1:]
        v = get(path)
        pool[key] = v
        st[key] = float(roc_auc_score(y, v))
    # Seed averages for T095
    for ep in (4, 6, 8):
        ks = [f't095_q_tabm-s{s}-ep{ep}' for s in (42, 43) if f't095_q_tabm-s{s}-ep{ep}' in pool]
        if len(ks) == 2:
            pool[f't095_q_tabm_2s_ep{ep}'] = 0.5 * (pool[ks[0]] + pool[ks[1]])
            st[f't095_q_tabm_2s_ep{ep}'] = float(roc_auc_score(y, pool[f't095_q_tabm_2s_ep{ep}']))
    for ep in (2, 3, 4):
        ks = [f't095_q_moe4-s{s}-ep{ep}' for s in (42, 43) if f't095_q_moe4-s{s}-ep{ep}' in pool]
        if len(ks) == 2:
            pool[f't095_q_moe4_2s_ep{ep}'] = 0.5 * (pool[ks[0]] + pool[ks[1]])
            st[f't095_q_moe4_2s_ep{ep}'] = float(roc_auc_score(y, pool[f't095_q_moe4_2s_ep{ep}']))
    return y, s024, pool, st

ye, s024e, pe, ste = load_pool('earlier')
yp, s024p, pp, stp = load_pool('primary')
names = [k for k in pe if k in pp]
Ze = np.column_stack([logit(np.clip(pe[k], 1e-6, 1 - 1e-6)) for k in names])
Zp = np.column_stack([logit(np.clip(pp[k], 1e-6, 1 - 1e-6)) for k in names])

base_e, base_p = float(roc_auc_score(ye, s024e)), float(roc_auc_score(yp, s024p))
counts = np.zeros(len(names), dtype=int); counts[0] = 5
cur_e = 5 * Ze[:, 0].copy(); cur_p = 5 * Zp[:, 0].copy()
trace = []
for step in range(1, 21):
    best_j, best_sc = -1, -1.0
    for j in range(len(names)):
        sc = float(roc_auc_score(ye, (cur_e + Ze[:, j]) / (counts.sum() + 1)))
        if sc > best_sc:
            best_sc, best_j = sc, j
    counts[best_j] += 1; cur_e += Ze[:, best_j]; cur_p += Zp[:, best_j]
    p_sc = float(roc_auc_score(yp, cur_p / counts.sum()))
    trace.append({'step': step, 'added': names[best_j], 'earlier': best_sc, 'earlier_delta': best_sc - base_e, 'primary': p_sc, 'primary_delta': p_sc - base_p})
    print(f"Step {step:2d} +{names[best_j]:22s}: earlier={best_sc:.6f} ({best_sc-base_e:+.6f}) | primary={p_sc:.6f} ({p_sc-base_p:+.6f})")

best_step = max(trace, key=lambda x: x['earlier'])
report = {
    'earlier_s024': base_e, 'primary_s024': base_p,
    't095_standalone_earlier': ste, 't095_standalone_primary': stp,
    'best_earlier_step': best_step, 'trace': trace,
    'final_weights': {names[j]: round(float(counts[j] / counts.sum()), 4) for j in range(len(names)) if counts[j] > 0},
}
(t / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
print('Best earlier step:', best_step)
