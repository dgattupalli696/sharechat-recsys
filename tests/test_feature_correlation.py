import unittest

import numpy as np

from feature_correlation import Moments, pairwise_correlations, target_correlations


class CorrelationTests(unittest.TestCase):
    def test_pairwise_observed_correlations_match_direct_reference(self):
        array = np.array([
            [1, 8, 1, np.nan], [2, np.nan, 1, np.nan],
            [3, 5, 1, np.nan], [4, 3, 1, np.nan],
            [np.inf, 2, 1, np.nan], [6, 1, 1, np.nan],
        ], dtype=np.float32)
        stats = Moments(4)
        stats.update(array[:2])
        stats.update(array[2:])
        correlation, counts = pairwise_correlations([array[:3], array[3:]], stats)
        for i in range(2):
            for j in range(2):
                mask = np.isfinite(array[:, i]) & np.isfinite(array[:, j])
                expected = np.corrcoef(array[mask, i], array[mask, j])[0, 1]
                self.assertAlmostEqual(correlation[i, j], expected, places=6)
                self.assertEqual(counts[i, j], mask.sum())
        self.assertTrue(np.isnan(correlation[2:]).all())

    def test_streaming_target_correlation_excludes_nonfinite_values(self):
        array = np.array([[1, 2], [4, np.nan], [2, 4], [8, 3], [5, 1], [6, 7]], dtype=np.float32)
        labels = np.array([0, 1, 0, 1, 0, 1])
        negative, positive = Moments(2), Moments(2)
        negative.update(array[labels == 0])
        positive.update(array[labels == 1])
        observed = target_correlations(negative, positive)
        for i in range(2):
            finite = np.isfinite(array[:, i])
            expected = np.corrcoef(array[finite, i], labels[finite])[0, 1]
            self.assertAlmostEqual(observed[i], expected, places=12)


if __name__ == "__main__":
    unittest.main()
