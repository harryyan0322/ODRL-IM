import unittest

import numpy as np
import torch

from partition_inference import (
    allocate_partition_quotas,
    partition_graph,
    select_partitioned_seeds,
)


class FeaturePolicy(torch.nn.Module):
    def forward(self, obs, edge_index):
        return obs[:, :1]


class PartitionInferenceTest(unittest.TestCase):
    def setUp(self):
        edges = [
            (0, 1), (1, 0), (1, 2), (2, 1), (2, 0), (0, 2),
            (3, 4), (4, 3), (4, 5), (5, 4), (5, 3), (3, 5),
            (2, 3), (3, 2),
        ]
        self.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    def test_leiden_partition_is_deterministic(self):
        first, _, backend = partition_graph(self.edge_index, 6, 2, seed=7)
        second, _, _ = partition_graph(self.edge_index, 6, 2, seed=7)
        self.assertEqual(backend, "leiden_rb_configuration")
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(set(first.tolist())), 2)

    def test_quota_allocation_preserves_partition_coverage(self):
        quotas, _ = allocate_partition_quotas(
            [np.array([3.0, 2.0]), np.array([1.0, 0.0])], [2, 2], 3
        )
        self.assertEqual(quotas.tolist(), [2, 1])

    def test_partitioned_inference_returns_unique_budget(self):
        obs = torch.tensor([[6.0], [5.0], [4.0], [3.0], [2.0], [1.0]])
        result = select_partitioned_seeds(
            FeaturePolicy(), obs, self.edge_index, budget=4,
            num_partitions=2, partition_seed=7
        )
        self.assertEqual(len(result.seeds), 4)
        self.assertEqual(len(set(result.seeds)), 4)
        self.assertEqual(sum(result.partition_quotas), 4)
        self.assertTrue(all(quota >= 1 for quota in result.partition_quotas))
        self.assertEqual(result.partition_backend, "leiden_rb_configuration")

    def test_global_topk_remains_explicit_baseline(self):
        obs = torch.tensor([[1.0], [3.0], [2.0], [6.0], [5.0], [4.0]])
        result = select_partitioned_seeds(
            FeaturePolicy(), obs, self.edge_index, budget=2,
            selection_mode="global_topk"
        )
        self.assertEqual(result.seeds, [3, 4])


if __name__ == "__main__":
    unittest.main()
