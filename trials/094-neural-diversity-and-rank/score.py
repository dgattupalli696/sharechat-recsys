"""Score T094 + full Caruana ensemble across S024 + T090 + T092 + T093 + T094 on both folds."""
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
        't069_d6_900': get(art / f'kaggle-t069/t069/{fold}-1-900.npz'),
        't090_tabm6': np.mean([get(art / f'kaggle-t090/t090/{fold}-tabm-w{w}-s{s}-ep6.npz') for w in (128, 256) for s in (42, 43)], axis=0),
        't090_dcn2': get(art / f'kaggle-t090/t090/{fold}-dcnv2-w128-s42-ep2.npz'),
        't090_dcn4': get(art / f'kaggle-t090/t090/{fold}-dcnv2-w128-s42-ep4.npz'),
        't092_d5_2000': get(art / f'kaggle-t092/t092/{fold}-xgb_d5-r2000.npz'),
        't092_d6_900': get(art / f'kaggle-t092/t092/{fold}-xgb_d6-r900.npz'),
        't092_d6_1200': get(art / f'kaggle-t092/t092/{fold}-xgb_d6-r1200.npz'),
        't092_cat6': get(art / f'kaggle-t092/t092/{fold}-cat_d6.npz'),
        't093_d4_stat': get(art / f'kaggle-t093/t093/{fold}-xgb_d4_stat-r3200.npz'),
    }
    t094_standalone = {}
    for path in sorted(folder.glob(f'{fold}-*.npz')):
        key = 't094_' + path.stem[len(fold) + 1:]
        v = get(path)
        pool[key] = v
        t094_standalone[key] = float(roc_auc_score(y, v))
    return y, s024, pool, t094_standalone

ye, s024e, pe, st_e = load_pool('earlier')
yp, s024p, pp, st_p = load_pool('primary')
names = [k for k in pe if k in pp]
Ze = np.column_stack([logit(np.clip(pe[k], 1e-6, 1 - 1e-6)) for k in names])
Zp = np.column_stack([logit(np.clip(pp[k], 1e-6, 1 - 1e-6)) for k in names])

base_e = float(roc_auc_score(ye, s024e))
base_p = float(roc_auc_score(yp, s024p))
counts = np.zeros(len(names), dtype=int)
counts[0] = 5
cur_e = 5 * Ze[:, 0].copy()
cur_p = 5 * Zp[:, 0].copy()
trace = []
for step in range(1, 21):
    best_j, best_sc = -1, -1.0
    for j in range(len(names)):
        sc = float(roc_auc_score(ye, (cur_e + Ze[:, j]) / (counts.sum() + 1)))
        if sc > best_sc:
            best_sc, best_j = sc, j
    counts[best_j] += 1
    cur_e += Ze[:, best_j]
    cur_p += Zp[:, best_j]
    p_sc = float(roc_auc_score(yp, cur_p / counts.sum()))
    trace.append({'step': step, 'added': names[best_j], 'earlier': best_sc, 'earlier_delta': best_sc - base_e, 'primary': p_sc, 'primary_delta': p_sc - base_p})
    print(f"Step {step:2d} +{names[best_j]:22s}: earlier={best_sc:.6f} ({best_sc-base_e:+.6f}) | primary={p_sc:.6f} ({p_sc-base_p:+.6f})")

best_step = max(trace, key=lambda x: x['earlier'])
report = {
    'earlier_s024': base_e, 'primary_s024': base_p,
    't094_standalone_earlier': st_e, 't094_standalone_primary': st_p,
    'best_earlier_step': best_step, 'trace': trace,
    'final_weights': {names[j]: round(float(counts[j] / counts.sum()), 4) for j in range(len(names)) if counts[j] > 0},
}
(t / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
print('Best earlier step:', best_step)
