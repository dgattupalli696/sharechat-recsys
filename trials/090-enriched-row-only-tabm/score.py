"""Score T090 enriched TabM/DCNv2 models against S024 (T086) on both chronological folds."""
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

specs = [
    ('tabm', 128, 42, (4, 6, 8, 10)),
    ('tabm', 128, 43, (4, 6, 8, 10)),
    ('tabm', 256, 42, (4, 6, 8, 10)),
    ('tabm', 256, 43, (4, 6, 8, 10)),
    ('dcnv2', 128, 42, (2, 4, 6)),
]

report = {}
for fold in ['earlier', 'primary']:
    ids, y, anchor, p = m['fold'](fold)
    def get(tr, k):
        i, l, v = m['read'](art / f'kaggle-t{tr}/t{tr}/{fold}-{k}.npz')
        np.testing.assert_array_equal(ids, i)
        np.testing.assert_array_equal(y, l)
        return v
    q = np.column_stack([
        anchor, *[p[k] for k in m['trees']],
        get('069', '1-2400'), get('070', 'gated-8'), get('072', 'xgb-base'),
    ])
    s024 = blend_s024(q, get('081', 'quantile-8'), get('083', 'paired'), get('086', 'learned'), get('086', 'random'))
    s024_auc = float(roc_auc_score(y, s024))
    standalone = {}
    preds = {}
    for arch, w, seed, eps in specs:
        for ep in eps:
            key = f'{arch}-w{w}-s{seed}-ep{ep}'
            path = folder / f'{fold}-{key}.npz'
            if not path.exists():
                continue
            i, l, v = m['read'](path)
            np.testing.assert_array_equal(ids, i)
            np.testing.assert_array_equal(y, l)
            preds[key] = v
            standalone[key] = float(roc_auc_score(y, v))
    # Build natural family averages
    for ep in (6, 8, 10):
        ks = [f'tabm-w{w}-s{s}-ep{ep}' for w in (128, 256) for s in (42, 43) if f'tabm-w{w}-s{s}-ep{ep}' in preds]
        if len(ks) >= 2:
            preds[f'tabm-all-ep{ep}'] = np.mean([preds[k] for k in ks], axis=0)
            standalone[f'tabm-all-ep{ep}'] = float(roc_auc_score(y, preds[f'tabm-all-ep{ep}']))
            for d_ep in (2, 4, 6):
                dk = f'dcnv2-w128-s42-ep{d_ep}'
                if dk in preds:
                    mix = 0.8 * preds[f'tabm-all-ep{ep}'] + 0.2 * preds[dk]
                    preds[f'tabm{ep}+dcn{d_ep}'] = mix
                    standalone[f'tabm{ep}+dcn{d_ep}'] = float(roc_auc_score(y, mix))
    grid = {}
    z_base = logit(np.clip(s024, 1e-8, 1 - 1e-8))
    for k, v in preds.items():
        z_v = logit(np.clip(v, 1e-8, 1 - 1e-8))
        for wt in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50):
            s_prob = (1 - wt) * s024 + wt * v
            s_logit = expit((1 - wt) * z_base + wt * z_v)
            grid[f'{k}/prob/{wt}'] = float(roc_auc_score(y, s_prob))
            grid[f'{k}/logit/{wt}'] = float(roc_auc_score(y, s_logit))
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

(t / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
for fold in ['earlier', 'primary']:
    r = report[fold]
    print(f"{fold}: S024={r['s024_auc']:.6f} -> {r['selected']} = {r['blend_auc']:.6f} (delta={r['delta_vs_s024']:+.6f})")
    best_s = max(r['standalone'].items(), key=lambda kv: kv[1])
    print(f"  best standalone: {best_s[0]} = {best_s[1]:.6f}")
