"""Turn externally generated seed sets into MPGNN compact offline data.

Learned behavior policies such as DeepIM, GCOMB, and PIANO can export a JSON
array of seed-index arrays. This adapter applies the same feature, WC-IC, and
Monte Carlo protocol as ``generate_offline_data.py`` without distributing any
baseline implementation.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "MPGNN"
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

from influence_evaluator import compute_mc_spread, read_weighted_cascade_graph
from offline_data import save_compact_pickle
from paper_protocol import bidirected_edges, pdf_node_features


def load_seed_sets(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("seed_sets")
    if not isinstance(value, list) or not value:
        raise ValueError("Input must be a non-empty JSON array or {seed_sets: [...]}")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-path", required=True)
    parser.add_argument("--seed-sets", required=True, help="JSON array of node-index arrays")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--budget-percent", type=float, required=True)
    parser.add_argument("--mc-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--node-index-base", type=int, default=0)
    parser.add_argument("--normalize-reward", action="store_true")
    args = parser.parse_args()
    if args.mc_trials <= 0:
        parser.error("--mc-trials must be positive")

    graph_path = Path(args.graph_path).resolve()
    num_nodes, edges = bidirected_edges(graph_path, args.node_index_base)
    features, _ = pdf_node_features(graph_path, args.node_index_base)
    graph = read_weighted_cascade_graph(graph_path, args.node_index_base, directed=False)
    budget_nodes = max(1, int(args.budget_percent * num_nodes / 100.0))
    seed_sets = []
    rewards = []
    for index, raw in enumerate(load_seed_sets(args.seed_sets)):
        selected = np.asarray(raw, dtype=np.int64).reshape(-1)
        if selected.size != budget_nodes:
            raise ValueError(f"seed set {index} has {selected.size} nodes; expected {budget_nodes}")
        if np.unique(selected).size != selected.size:
            raise ValueError(f"seed set {index} contains duplicate nodes")
        if selected.min() < 0 or selected.max() >= num_nodes:
            raise ValueError(f"seed set {index} contains an out-of-range node")
        rng = np.random.default_rng(args.seed * 1000003 + index)
        reward = compute_mc_spread(graph, selected, args.mc_trials, rng)
        if args.normalize_reward:
            reward /= num_nodes
        seed_sets.append(np.sort(selected).tolist())
        rewards.append(float(reward))

    adjacency = sp.csr_matrix(
        (np.ones(len(edges), dtype=np.float32),
         ([source for source, _ in edges], [target for _, target in edges])),
        shape=(num_nodes, num_nodes),
    )
    initial_obs = features.copy()
    initial_obs[:, 3] = 0.0
    output = Path(args.output).resolve()
    save_compact_pickle(initial_obs, seed_sets, rewards, adjacency, output)
    metadata = {
        "format": "mpgnn_compact_v1",
        "protocol_id": "r2_pdf_locked_v1",
        "dataset": args.dataset,
        "graph_path": str(graph_path),
        "seed_set_source": str(Path(args.seed_sets).resolve()),
        "num_nodes": num_nodes,
        "num_samples": len(seed_sets),
        "budget_percent": args.budget_percent,
        "budget_nodes": budget_nodes,
        "seed": args.seed,
        "diffusion_model": "weighted_cascade_ic",
        "collection_mc_trials": args.mc_trials,
        "evaluation_mc_trials": 1000,
        "graph_mode": "bidirected_from_undirected_edge_list",
        "node_features": ["in_degree", "out_degree", "pagerank", "seed_availability"],
        "normalize_reward": bool(args.normalize_reward),
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"saved {output}")


if __name__ == "__main__":
    main()
