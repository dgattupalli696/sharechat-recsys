"""Score T093 shallow long-horizon & pairwise-rank XGBoost + full Caruana ensemble on both folds."""
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

report = {}
for fold in ['earlier', 'primary']:
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
    s024_auc = float(roc_auc_score(y, s024))
    preds, standalone = {}, {}
    for path in sorted(folder.glob(f'{fold}-*.npz')):
        key = path.stem[len(fold) + 1:]
        v = get(path)
        preds[key] = v
        standalone[key] = float(roc_auc_score(y, v))
    grid = {}
    z_base = logit(np.clip(s024, 1e-8, 1 - 1e-8))
    for k, v in preds.items():
        z_v = logit(np.clip(v, 1e-8, 1 - 1e-8))
        for wt in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
            grid[f'{k}/logit/{wt}'] = float(roc_auc_score(y, expit((1 - wt) * z_base + wt * z_v)))
            grid[f'{k}/prob/{wt}'] = float(roc_auc_score(y, (1 - wt) * s024 + wt * v))
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
    for k, v in sorted(r['standalone'].items(), key=lambda kv: -kv[1])[:6]:
        print(f"  standalone {k:18s}: {v:.6f}")
