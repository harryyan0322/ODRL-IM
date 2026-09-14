"""Generate compact offline transitions using the locked MPGNN protocol.

The script is independent of baseline implementations. It provides a
reproducible public fallback for random, degree, and PageRank behavior
mixtures; learned policies can export seed sets and use the same compact
format.
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


def parse_strategies(value):
    names, weights = [], []
    for item in value.split(","):
        name, weight = item.split(":", 1)
        name = name.strip().lower()
        weight = float(weight)
        if name not in {"random", "degree", "pagerank"}:
            raise ValueError(f"Unsupported strategy: {name}")
        if weight <= 0:
            raise ValueError("Strategy weights must be positive")
        names.append(name)
        weights.append(weight)
    if len(set(names)) != len(names):
        raise ValueError("Each strategy may occur only once")
    weights = np.asarray(weights, dtype=np.float64)
    return names, weights / weights.sum()


def weighted_schedule(names, weights, count):
    used = np.zeros(len(names), dtype=np.int64)
    schedule = []
    for step in range(int(count)):
        index = int(np.argmax((step + 1) * weights - used))
        used[index] += 1
        schedule.append(names[index])
    return schedule, dict(zip(names, used.tolist()))


def select_seeds(strategy, scores, budget_nodes, rng):
    num_nodes = scores.size
    budget_nodes = min(max(int(budget_nodes), 1), num_nodes)
    if strategy == "random":
        return np.sort(rng.choice(num_nodes, size=budget_nodes, replace=False))
    # Stable jitter makes ties reproducible without systematically favoring
    # low node identifiers.
    order = np.argsort(
        -(scores + rng.random(num_nodes) * 1e-9), kind="mergesort"
    )
    return np.sort(order[:budget_nodes])


def generate(args):
    graph_path = Path(args.graph_path).resolve()
    output = Path(args.output).resolve()
    num_nodes, edges = bidirected_edges(graph_path, args.node_index_base)
    features, _ = pdf_node_features(graph_path, args.node_index_base)
    graph = read_weighted_cascade_graph(
        graph_path, node_index_base=args.node_index_base, directed=False
    )
    if graph.num_nodes != num_nodes:
        raise ValueError("Graph protocol loaders disagree on node count")

    sources = np.asarray([source for source, _ in edges], dtype=np.int64)
    degree = np.bincount(sources, minlength=num_nodes).astype(np.float64)
    score_map = {"degree": degree, "pagerank": features[:, 2].astype(np.float64)}
    names, weights = parse_strategies(args.strategies)
    schedule, counts = weighted_schedule(names, weights, args.num_samples)
    rng = np.random.default_rng(args.seed)
    budget_nodes = max(1, int(float(args.budget_percent) * num_nodes / 100.0))

    seed_sets, rewards = [], []
    for index, strategy in enumerate(schedule):
        scores = np.zeros(num_nodes) if strategy == "random" else score_map[strategy]
        selected = select_seeds(strategy, scores, budget_nodes, rng)
        reward_rng = np.random.default_rng(int(args.seed) * 1000003 + index)
        reward = compute_mc_spread(graph, selected, args.mc_trials, reward_rng)
        if args.normalize_reward:
            reward /= num_nodes
        seed_sets.append(selected.tolist())
        rewards.append(float(reward))
        if args.progress_every and (index + 1) % args.progress_every == 0:
            print(f"generated {index + 1}/{args.num_samples}", flush=True)

    adjacency = sp.csr_matrix(
        (np.ones(len(edges), dtype=np.float32),
         ([source for source, _ in edges], [target for _, target in edges])),
        shape=(num_nodes, num_nodes),
    )
    initial_obs = features.copy()
    initial_obs[:, 3] = 0.0
    save_compact_pickle(initial_obs, seed_sets, rewards, adjacency, output)
    metadata = {
        "format": "mpgnn_compact_v1",
        "protocol_id": "r2_pdf_locked_v1",
        "dataset": args.dataset,
        "graph_path": str(graph_path),
        "num_nodes": num_nodes,
        "num_samples": int(args.num_samples),
        "budget_percent": float(args.budget_percent),
        "budget_nodes": budget_nodes,
        "strategies": [
            {"name": name, "weight": float(weight), "count": counts[name]}
            for name, weight in zip(names, weights)
        ],
        "seed": int(args.seed),
        "diffusion_model": "weighted_cascade_ic",
        "collection_mc_trials": int(args.mc_trials),
        "evaluation_mc_trials": 1000,
        "graph_mode": "bidirected_from_undirected_edge_list",
        "node_features": ["in_degree", "out_degree", "pagerank", "seed_availability"],
        "normalize_reward": bool(args.normalize_reward),
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"saved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--budget-percent", type=float, default=10.0)
    parser.add_argument("--strategies", default="random:0.2,degree:0.4,pagerank:0.4")
    parser.add_argument("--mc-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--node-index-base", type=int, default=0)
    parser.add_argument("--normalize-reward", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()
    if args.num_samples <= 0 or args.mc_trials <= 0:
        parser.error("--num-samples and --mc-trials must be positive")
    generate(args)


if __name__ == "__main__":
    main()
