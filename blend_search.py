"""Exhaustive equal-weight rank-blend search over every model with valid holdout predictions.

Only predictions from models fit on the pre-cutoff rows and scored out-of-sample on the
original 307,528-row holdout are used. Full-data refits, k-fold/OOF, test predictions,
in-sample replays (T022), saved blends/stacks and holdout-label cross-fits (T033) are excluded.

Search: (1) every subset up to --max-size over all models; (2) every subset of every size
over a --pool of the strongest models (greedy diversity pruning). Subsets are enumerated
in Gray-code order so each step adds or removes one rank vector, and AUC is computed with
an O(n) histogram over the averaged ranks; the best --top results are re-scored exactly.
"""
import argparse
from itertools import combinations
import json
from multiprocessing import Pool
from pathlib import Path
import re
import time

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

HOLDOUT_ROWS = 307_528
EXCLUDE = re.compile(r"(test|kfold|oof|refit|former-holdout|equal_weight|preliminary-blend|frozen-|"
                     r"entity-coverage|evaluation-reference|graph-data|local-t033|local-t020|"
                     r"local-t01[3567]/validation_predictions|local-t017-stacking/validation_predictions)")
NOT_PREDICTION = {"row_ids", "rows", "y_true", "ensemble", "left_seen", "right_seen"}
BINS = 1 << 20


def discover(root):
    found = {}
    for path in sorted(Path(root).glob("**/*.npz")):
        rel = str(path.relative_to(root))
        if EXCLUDE.search(rel):
            continue
        z = np.load(path)
        if "row_ids" not in z.files or z["row_ids"].shape != (HOLDOUT_ROWS,):
            continue
        name = re.sub(r"(validation_predictions|-holdout|holdout|validation-replay|validation-reference)", "",
                      rel.removesuffix(".npz"))
        name = re.sub(r"^kaggle-(t\d+)([^/]*)/\1/", r"\1\2/", name).strip("/-").replace("//", "/")
        keys = [k for k in z.files if k not in NOT_PREDICTION and z[k].shape == (HOLDOUT_ROWS,)]
        if {"p1", "p2"} <= set(keys):
            found[f"{name}/hurdle"] = (z["row_ids"], z["p1"] * z["p2"], z.get("y_true"))
            continue
        for k in keys:
            found[name if k == "probabilities" else f"{name}/{k}"] = (z["row_ids"], z[k], z.get("y_true"))
    return found


def align(found):
    ref_ids = None
    for ids, _, y in found.values():
        if y is not None:
            ref_ids, y_ref = ids, y
            break
    order = np.argsort(ref_ids)
    names, ranks = [], []
    for name, (ids, p, y) in found.items():
        pos = np.searchsorted(ref_ids, ids, sorter=order)
        idx = order[np.clip(pos, 0, len(order) - 1)]
        if not np.array_equal(ref_ids[idx], ids):
            print(f"skip {name}: row_ids differ from the reference holdout")
            continue
        if y is not None and not np.array_equal(y_ref[idx], y):
            print(f"skip {name}: labels disagree")
            continue
        aligned = np.empty(HOLDOUT_ROWS, dtype=np.float64)
        aligned[idx] = p
        names.append(name)
        ranks.append(rankdata(aligned) / HOLDOUT_ROWS)
    return names, np.asarray(ranks, dtype=np.float32), y_ref.astype(np.int8)


def histogram_auc(score, k, y):
    """AUC of score/k (in (0, 1]) via BINS-bucket histograms; ties within a bucket count half."""
    b = np.minimum((score * (BINS / k)).astype(np.int64), BINS - 1)
    pos = np.bincount(b[y == 1], minlength=BINS)
    neg = np.bincount(b[y == 0], minlength=BINS)
    below = np.cumsum(neg) - neg
    return float((pos * (below + 0.5 * neg)).sum() / (pos.sum() * neg.sum()))


_R = _Y = None


def _init(ranks, y):
    global _R, _Y
    _R, _Y = ranks, y


def _gray_chunk(job):
    """Enumerate all non-empty subsets of `pool` with the fixed high bits `prefix`."""
    pool, prefix_bits, low = job
    members = [pool[i] for i in range(len(prefix_bits)) if prefix_bits[i]]
    lows = pool[len(prefix_bits):][:low]
    score = _R[members].sum(0, dtype=np.float32) if members else np.zeros(_R.shape[1], np.float32)
    active, best = set(), []
    for g in range(1 << len(lows)):
        if g:
            bit = (g & -g).bit_length() - 1
            m = lows[bit]
            if m in active:
                active.remove(m)
                score -= _R[m]
            else:
                active.add(m)
                score += _R[m]
        k = len(members) + len(active)
        if k:
            best.append((histogram_auc(score, k, _Y), tuple(sorted(members + list(active)))))
    best.sort(reverse=True)
    return best[:200]


def _subset_chunk(subsets):
    out = [(histogram_auc(_R[list(s)].sum(0, dtype=np.float32), len(s), _Y), s) for s in subsets]
    out.sort(reverse=True)
    return out[:200]


def select_pool(names, ranks, y, singles, size, max_corr):
    """Greedy: strongest models first, skipping any with rank correlation > max_corr to a chosen one."""
    chosen = []
    for i in np.argsort(singles)[::-1]:
        if all(np.corrcoef(ranks[i], ranks[j])[0, 1] <= max_corr for j in chosen):
            chosen.append(int(i))
        if len(chosen) == size:
            break
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--max-size", type=int, default=3, help="exhaustive subset size over all models")
    parser.add_argument("--pool", type=int, default=18, help="models searched over all 2^pool subsets")
    parser.add_argument("--max-corr", type=float, default=0.995, help="pool diversity threshold")
    parser.add_argument("--min-auc", type=float, default=0.0, help="drop models below this single AUC")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path("artifacts/local-blend-search"))
    args = parser.parse_args()
    began = time.time()

    names, ranks, y = align(discover(args.artifacts))
    keep, seen = [], []
    for i in range(len(names)):
        dup = next((names[j] for j in seen if np.corrcoef(ranks[i], ranks[j])[0, 1] > 0.99999), None)
        if dup:
            print(f"duplicate {names[i]} == {dup}")
        else:
            seen.append(i)
            keep.append(i)
    names, ranks = [names[i] for i in keep], ranks[keep]
    singles = np.array([roc_auc_score(y, r) for r in ranks])
    keep = np.flatnonzero(singles >= args.min_auc)
    names, ranks, singles = [names[i] for i in keep], ranks[keep], singles[keep]
    print(f"{len(names)} models ({time.time() - began:.0f}s)")
    for i in np.argsort(singles)[::-1]:
        print(f"  {singles[i]:.5f}  {names[i]}")

    results = {}
    with Pool(args.workers, initializer=_init, initargs=(ranks, y)) as pool:
        subsets = [s for k in range(1, args.max_size + 1) for s in combinations(range(len(names)), k)]
        print(f"phase 1: {len(subsets):,} subsets of size <= {args.max_size}", flush=True)
        chunks = [subsets[i::args.workers * 8] for i in range(args.workers * 8)]
        for part in pool.imap_unordered(_subset_chunk, chunks):
            results.update({s: a for a, s in part})
        print(f"  done ({time.time() - began:.0f}s)", flush=True)

        chosen = select_pool(names, ranks, y, singles, args.pool, args.max_corr)
        print(f"phase 2: all {2 ** len(chosen) - 1:,} subsets of pool: {[names[i] for i in chosen]}", flush=True)
        high = min(6, len(chosen))
        jobs = [(chosen, [(h >> b) & 1 for b in range(high)], len(chosen) - high) for h in range(1 << high)]
        for part in pool.imap_unordered(_gray_chunk, jobs):
            results.update({s: a for a, s in part})
        print(f"  done ({time.time() - began:.0f}s)", flush=True)

    ranked = sorted(results.items(), key=lambda t: -t[1])[: args.top * 3]
    exact = sorted(((roc_auc_score(y, ranks[list(s)].mean(0)), s) for s, _ in ranked), reverse=True)[: args.top]
    print(f"\ntop {args.top} equal-weight rank blends (exact AUC):")
    for auc, s in exact:
        print(f"  {auc:.5f}  k={len(s):2d}  " + " + ".join(names[i] for i in s))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps({
        "models": {n: float(a) for n, a in zip(names, singles)},
        "pool": [names[i] for i in chosen],
        "top": [{"auc": float(a), "models": [names[i] for i in s]} for a, s in exact],
    }, indent=1))


if __name__ == "__main__":
    main()
