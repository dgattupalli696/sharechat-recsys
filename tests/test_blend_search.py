import numpy as np
from sklearn.metrics import roc_auc_score

import blend_search as bs


def test_histogram_auc_matches_sklearn_for_rank_means():
    rng = np.random.default_rng(0)
    y = (rng.random(20_000) < 0.1).astype(np.int8)
    ranks = np.stack([np.argsort(np.argsort(rng.random(20_000) + y * w)) + 1 for w in (0.3, 0.5, 0.8)]) / 20_000
    score = ranks.sum(0)
    assert abs(bs.histogram_auc(score, 3, y) - roc_auc_score(y, score)) < 1e-4


def test_gray_chunk_scores_every_subset_once():
    rng = np.random.default_rng(1)
    y = (rng.random(5_000) < 0.2).astype(np.int8)
    ranks = (rng.random((5, 5_000)) + y * np.linspace(0.1, 0.5, 5)[:, None]).astype(np.float32)
    ranks = np.argsort(np.argsort(ranks, 1), 1).astype(np.float32) / 5_000 + 1 / 5_000
    bs._init(ranks, y)
    pool = [4, 2, 0, 3, 1]
    out = [r for h in range(4) for r in bs._gray_chunk((pool, [(h >> b) & 1 for b in range(2)], 3))]
    subsets = [s for _, s in out]
    assert len(subsets) == len(set(subsets)) == 31
    for auc, s in out:
        assert abs(auc - roc_auc_score(y, ranks[list(s)].sum(0))) < 1e-3
