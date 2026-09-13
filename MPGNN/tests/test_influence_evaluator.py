import tempfile
import unittest
from pathlib import Path

import numpy as np

from influence_evaluator import (
    compute_mc_spread,
    read_weighted_cascade_graph,
    seed_dispersion_metrics,
    simulate_ic_spreads,
)


class InfluenceEvaluatorTest(unittest.TestCase):
    def test_weighted_cascade_chain(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph.txt"
            path.write_text("0 1\n1 2\n", encoding="utf-8")
            graph = read_weighted_cascade_graph(path, directed=True)
            spread = compute_mc_spread(
                graph, [0], num_trials=10, rng=np.random.default_rng(1)
            )
            self.assertEqual(spread, 3.0)

    def test_graph_ids_must_be_consecutive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph.txt"
            path.write_text("0 2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_weighted_cascade_graph(path, directed=True)

    def test_simulation_returns_each_trial(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "graph.txt"
            path.write_text("0 1\n", encoding="utf-8")
            graph = read_weighted_cascade_graph(path, directed=True)
            spreads = simulate_ic_spreads(
                graph, [0], num_trials=7, rng=np.random.default_rng(3)
            )
            self.assertEqual(spreads.shape, (7,))
            self.assertTrue(np.all(spreads == 2))


if __name__ == "__main__":
    unittest.main()
