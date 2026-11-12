import unittest

import torch
from torch.nn import functional as F

from feature_models import FeatureEncoder, HeterogeneousClassifier


class FeatureModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.numeric = torch.randn(8, 70)
        vectors = self.numeric[:, 2:].reshape(8, 2, 34)
        vectors[:, :, :32] = F.normalize(vectors[:, :, :32], dim=-1)
        vectors[:, :, 32] = 0.5
        vectors[:, :, 33] = 1
        vectors[0, 0] = 0
        self.categories = torch.ones(8, 1, dtype=torch.int64)
        self.config = {
            "n_numeric": 70, "n_dense": 2, "n_vectors": 2,
            "cardinalities": [3], "continuous_indices": [0, 1],
            "embedding_dimensions": [4], "summary_groups": [[0], [1]],
        }

    def test_all_encoders_and_backbones_have_finite_gradients(self):
        settings = [
            {"numerical_encoding": "ple", "bins": [[-5, 0, 5]] * 2},
            {"numerical_encoding": "plr"},
            {"vector_projection": True},
            {"vector_dropout": 0.1},
            {"vector_crosses": True},
        ]
        for extra in settings:
            for architecture, options in [
                ("tabm", {}), ("dcnv2", {}), ("dcnv2", {"experts": 4}),
                ("finalmlp", {}), ("finalmlp", {"gating": False}),
            ]:
                with self.subTest(extra=extra, architecture=architecture, options=options):
                    model = HeterogeneousClassifier(
                        {**self.config, **extra}, architecture=architecture,
                        width=8, members=2, **options,
                    )
                    logits = model(self.numeric, self.categories)
                    self.assertEqual(logits.shape, (8, 2 if architecture == "tabm" else 1))
                    target = torch.tensor([0.0, 1.0] * 4).unsqueeze(1).expand_as(logits)
                    F.binary_cross_entropy_with_logits(logits, target).backward()
                    self.assertTrue(torch.isfinite(logits).all().item())
                    self.assertTrue(all(
                        torch.isfinite(parameter.grad).all().item()
                        for parameter in model.parameters() if parameter.grad is not None
                    ))

    def test_whole_vector_dropout_also_masks_derived_scalar_summaries(self):
        encoder = FeatureEncoder(**self.config, vector_dropout=1.0)
        changed = self.numeric * 2
        encoder.train()
        torch.manual_seed(42)
        original = encoder(self.numeric, self.categories)
        torch.manual_seed(42)
        dropped = encoder(changed, self.categories)
        torch.testing.assert_close(original, dropped)

    def test_missing_vector_pairs_have_zero_interactions(self):
        encoder = FeatureEncoder(**self.config, vector_crosses=True)
        encoder.eval()
        features = encoder(self.numeric, self.categories)
        torch.testing.assert_close(features[0, -4:], torch.zeros(4))
        self.assertEqual(features[1, -1].item(), 1)


if __name__ == "__main__":
    unittest.main()
