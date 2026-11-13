import unittest

import numpy as np
import torch

from feed_attention import FeedAttentionClassifier, feed_similarity, request_batches, request_layout


def encoder_config(n_dense=3, n_vectors=2):
    return {
        "n_numeric": n_dense + 34 * n_vectors, "n_dense": n_dense, "n_vectors": n_vectors,
        "cardinalities": [5, 7], "continuous_indices": [0, 1], "embedding_dimensions": [4, 4],
        "numerical_encoding": "plr", "vector_crosses": True,
    }


def rows(count, n_dense=3, n_vectors=2, seed=0):
    generator = torch.Generator().manual_seed(seed)
    vectors = torch.randn(count, n_vectors, 32, generator=generator)
    vectors = torch.cat([
        vectors / vectors.norm(dim=-1, keepdim=True),
        torch.rand(count, n_vectors, 1, generator=generator),
        torch.ones(count, n_vectors, 1),
    ], dim=-1)
    numeric = torch.cat([torch.randn(count, n_dense, generator=generator), vectors.flatten(1)], dim=1)
    return numeric, torch.randint(0, 5, (count, 2), generator=generator)


class FeedAttentionTest(unittest.TestCase):
    def test_batches_keep_requests_whole_and_cover_every_row(self):
        ids = np.array(["b", "a", "b", "c", "a", "a", "d"])
        layout = request_layout(ids)
        seen = []
        for indices, groups in request_batches(layout, 3, torch.Generator().manual_seed(1)):
            batch_ids = ids[indices.numpy()]
            for group in groups.unique():
                members = set(batch_ids[(groups == group).numpy()])
                self.assertEqual(len(members), 1)
                self.assertEqual((ids == members.pop()).sum(), (groups == group).sum().item())
            self.assertTrue((groups[1:] >= groups[:-1]).all())
            seen.extend(indices.tolist())
        self.assertEqual(sorted(seen), list(range(len(ids))))

    def test_similarity_excludes_self_and_singletons(self):
        numeric, _ = rows(3)
        groups = torch.tensor([0, 0, 1])
        similarity = feed_similarity(numeric, groups, 3, 2)
        vectors = numeric[:, 3:].reshape(3, 2, 34)[:, :, :32]
        expected = (vectors[0] * vectors[1]).sum(-1)
        torch.testing.assert_close(similarity[0, :2], expected)
        self.assertTrue((similarity[2] == 0).all())

    def test_predictions_depend_only_on_own_request(self):
        torch.manual_seed(0)
        model = FeedAttentionClassifier(encoder_config(), width=16, members=2, heads=2, layers=1).eval()
        numeric, categorical = rows(6)
        with torch.no_grad():
            together = model(numeric, categorical, torch.tensor([0, 0, 1, 1, 1, 2]))
            alone = model(numeric[2:5], categorical[2:5], torch.tensor([0, 0, 0]))
            changed = numeric.clone()
            changed[0] += 5
            neighbour = model(changed, categorical, torch.tensor([0, 0, 1, 1, 1, 2]))
        torch.testing.assert_close(together[2:5], alone, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(neighbour[2:], together[2:], atol=1e-5, rtol=1e-5)
        self.assertFalse(torch.allclose(neighbour[1], together[1]))


if __name__ == "__main__":
    unittest.main()
