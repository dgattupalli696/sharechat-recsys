import unittest

import numpy as np
import polars as pl
import torch

from multitower import MultiTower, fit
from multitower_prepare import BINS, bin_matrix, id_codes, quantile_grid


class BinningTests(unittest.TestCase):
    def test_bins_are_monotone_and_missing_is_negative(self):
        x = np.array([[1.0, np.nan], [2.0, 5.0], [3.0, 6.0], [np.nan, 7.0]], dtype=np.float32)
        grid = quantile_grid(x, sample=10)
        codes = bin_matrix(x, grid, chunk=2)
        self.assertEqual(codes[3, 0], -1)
        self.assertEqual(codes[0, 1], -1)
        self.assertTrue(codes[0, 0] < codes[1, 0] < codes[2, 0] <= BINS)

    def test_id_codes_map_rare_and_unseen_to_zero(self):
        fit_frame = pl.DataFrame({"a": ["x", "x", "x", "y", None]})
        valid_frame = pl.DataFrame({"a": ["y", "x", "z"]})
        fit_codes, valid_codes, cards = id_codes(fit_frame, valid_frame, minimum=2)
        self.assertEqual(cards, [2])
        np.testing.assert_array_equal(fit_codes[:, 0], [1, 1, 1, 0, 0])
        np.testing.assert_array_equal(valid_codes[:, 0], [0, 1, 0])


class MultiTowerTests(unittest.TestCase):
    def data(self, n=4000, seed=0):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, 6)).astype(np.float32)
        x[rng.random((n, 6)) < 0.1] = np.nan
        ids = rng.integers(0, 20, size=(n, 2)).astype(np.int32)
        leaves = rng.integers(0, 8, size=(n, 5)).astype(np.int16)
        signal = np.nan_to_num(x[:, 0]) + (ids[:, 0] % 2) + (leaves[:, 1] == 3)
        y = (signal + rng.normal(size=n) > 1.5).astype(np.float32)
        bins = bin_matrix(x, quantile_grid(x))
        return [torch.from_numpy(bins), torch.from_numpy(ids), torch.from_numpy(leaves)], torch.from_numpy(y)

    def test_every_tower_subset_runs_forward_and_backward(self):
        tensors, y = self.data(64)
        for towers in [("content",), ("id",), ("leaf",), ("content", "id"), ("content", "id", "leaf")]:
            model = MultiTower(6, [20, 20], 5, 8, towers=towers, width=16, leaf_groups=2)
            model.train()
            logits, aux = model(*tensors)
            self.assertEqual(logits.shape, (64,))
            self.assertEqual(aux.shape, (64, len(towers)))
            (logits.sum() + aux.sum()).backward()

    def test_fit_learns_signal_from_all_towers(self):
        train, y_train = self.data(4000, 0)
        valid, y_valid = self.data(2000, 1)
        model = MultiTower(6, [20, 20], 5, 8, width=32, leaf_groups=5, prior=float(y_train.mean()))
        auc, pred, history = fit(model, train, y_train, valid, y_valid, epochs=4, batch=256, lr=3e-3,
                                 log=lambda s: None)
        self.assertGreater(auc, 0.75)
        self.assertEqual(pred.shape, (2000,))
        self.assertEqual(auc, max(h["auc"] for h in history))


if __name__ == "__main__":
    unittest.main()
