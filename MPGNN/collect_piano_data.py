"""Collect offline influence-maximization data from a pretrained PIANO policy."""

import argparse
import ctypes
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from collect_behavior_data import _parse_epsilon_mixture, _weighted_schedule
from influence_evaluator import compute_mc_spread, read_weighted_cascade_graph
from offline_data import load_pickle_dataset, save_compact_pickle, save_legacy_pickle
from paper_protocol import pdf_node_features
from behavior_policy_sources import BEHAVIOR_POLICY_SOURCES


def _read_edges(path, node_index_base, directed):
    edges = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith(("#", "%")):
                continue
            fields = line.split()
            source = int(fields[0]) - node_index_base
            target = int(fields[1]) - node_index_base
            # PIANO's released QNet counts both endpoints before inserting
            # them into its candidate map. A self-loop is therefore counted
            # twice and triggers qnet.cpp's availability assertion. Self-loops
            # carry no additional influence under IC and are removed here.
            if source == target:
                continue
            edges.append((source, target))
            if not directed:
                edges.append((target, source))
    return edges


class PianoPolicy:
    """Thin ctypes adapter around PIANO's native CPU inference library."""

    def __init__(self, library, model_file, edges, num_nodes, budget_nodes):
        self.library_path = str(Path(library).resolve())
        self.model_path = str(Path(model_file).resolve())
        self.num_nodes = int(num_nodes)
        self.lib = ctypes.CDLL(self.library_path)
        self.lib.Init.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        self.lib.InsertGraph.argtypes = [
            ctypes.c_bool,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        self.lib.LoadModel.argtypes = [ctypes.c_char_p]
        self.lib.GetSol_epsilon.argtypes = [
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_uint,
            ctypes.POINTER(ctypes.c_int),
        ]
        self.lib.GetSol_epsilon.restype = ctypes.c_double

        arguments = [
            "piano",
            "-max_bp_iter", "1",
            "-embed_dim", "64",
            "-reg_hidden", "64",
            "-seed_k", str(budget_nodes),
            "-max_n", str(num_nodes),
            "-num_env", "1",
            "-batch_size", "1",
        ]
        encoded = (ctypes.c_char_p * len(arguments))(
            *[argument.encode("utf-8") for argument in arguments]
        )
        self.lib.Init(len(arguments), encoded)

        edge_sources = (ctypes.c_int * len(edges))(*[edge[0] for edge in edges])
        edge_targets = (ctypes.c_int * len(edges))(*[edge[1] for edge in edges])
        self.lib.InsertGraph(
            True, 0, num_nodes, len(edges), edge_sources, edge_targets
        )
        self.lib.LoadModel(self.model_path.encode("utf-8"))

    def select(self, epsilon, seed):
        solution = (ctypes.c_int * (self.num_nodes + 10))()
        self.lib.GetSol_epsilon(0, float(epsilon), int(seed), solution)
        selected = [int(solution[index + 1]) for index in range(solution[0])]
        if len(selected) != len(set(selected)):
            raise RuntimeError("PIANO returned duplicate seed nodes")
        return selected


def _collect(args):
    template = load_pickle_dataset(args.template_data)
    num_nodes = np.asarray(template["obs"][0]).shape[0]
    budget_nodes = max(1, int(args.budget_percent * num_nodes / 100.0))
    # The released PIANO implementation trains and infers on an undirected
    # graph (its own main.py explicitly creates g_undirected). Preserve that
    # native policy input while evaluating the selected nodes on the directed
    # WC-IC graph used by the paper's main metric.
    edges = _read_edges(args.graph_path, args.node_index_base, False)
    evaluator_graph = read_weighted_cascade_graph(
        args.graph_path,
        node_index_base=args.node_index_base,
        directed=args.directed,
    )
    if evaluator_graph.num_nodes != num_nodes:
        raise ValueError("Graph and template node counts do not match")

    policy = PianoPolicy(
        args.library, args.model_file, edges, num_nodes, budget_nodes
    )
    mixture = _parse_epsilon_mixture(args.epsilon_mixture)
    epsilons, epsilon_counts = _weighted_schedule(
        [item[0] for item in mixture],
        [item[1] for item in mixture],
        args.num_samples,
    )

    result = {"obs": [], "next_obs": [], "actions": [], "rewards": []}
    pdf_features, pdf_edges = pdf_node_features(args.graph_path, args.node_index_base)
    seed_sets = []
    unique_seed_sets = set()
    for sample_index, epsilon in enumerate(epsilons):
        policy_seed = args.seed * 1000003 + sample_index
        selected = policy.select(epsilon, policy_seed)
        if len(selected) != budget_nodes:
            raise RuntimeError(
                f"PIANO returned {len(selected)} seeds, expected {budget_nodes}"
            )

        template_index = sample_index % len(template["obs"])
        obs = pdf_features.copy()
        obs[:, -1] = 0.0
        next_obs = obs.copy()
        next_obs[selected, -1] = 1.0
        action = np.zeros((num_nodes, 1), dtype=np.float32)
        action[selected, 0] = 1.0
        reward_rng = np.random.default_rng(policy_seed)
        reward = compute_mc_spread(
            evaluator_graph, selected, args.mc_trials, reward_rng
        )
        if args.normalize_reward:
            reward /= num_nodes

        result["obs"].append(obs)
        result["next_obs"].append(next_obs)
        result["actions"].append(action)
        result["rewards"].append(float(reward))
        seed_sets.append(selected)
        unique_seed_sets.add(tuple(sorted(selected)))
        if args.progress_every and (sample_index + 1) % args.progress_every == 0:
            print(f"Collected {sample_index + 1}/{args.num_samples}")

    result["adj_matrix"] = sp.csr_matrix(
        (np.ones(len(pdf_edges), dtype=np.float32),
         ([edge[0] for edge in pdf_edges], [edge[1] for edge in pdf_edges])),
        shape=(num_nodes, num_nodes),
    )
    if args.compact:
        save_compact_pickle(
            result["obs"][0], seed_sets, result["rewards"],
            result["adj_matrix"], args.output
        )
    else:
        save_legacy_pickle(result, args.output)

    metadata = {
        "source": "piano",
        "collector": "PIANO_native_cpu",
        "upstream_repository": BEHAVIOR_POLICY_SOURCES["PIANO"]["repository"],
        "library": policy.library_path,
        "model_file": policy.model_path,
        "template_data": str(Path(args.template_data).resolve()),
        "graph_path": str(Path(args.graph_path).resolve()),
        "num_samples": args.num_samples,
        "unique_seed_sets": len(unique_seed_sets),
        "budget_percent": args.budget_percent,
        "budget_nodes": budget_nodes,
        "epsilon_mixture": [
            {"epsilon": epsilon, "weight": weight, "count": count}
            for (epsilon, weight), count in zip(mixture, epsilon_counts)
        ],
        "exploration": "per_decision_epsilon_greedy",
        "diffusion_model": "weighted_cascade_ic",
        "mc_trials": args.mc_trials,
        "seed": args.seed,
        "action_encoding": "topk_k_hot",
        "directed": False,
        "protocol_id": "r2_pdf_locked_v1",
        "graph_mode": "bidirected_from_undirected_edge_list",
        "node_feature_dim": 4,
        "policy_graph_projection": "undirected",
        "node_index_base": args.node_index_base,
        "normalize_reward": args.normalize_reward,
        "storage_format": "mpgnn_compact_v1" if args.compact else "legacy",
    }
    metadata_path = Path(args.output).with_suffix(Path(args.output).suffix + ".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Saved PIANO source to {Path(args.output).resolve()}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--template-data", required=True)
    parser.add_argument("--graph-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--budget-percent", type=float, required=True)
    parser.add_argument(
        "--epsilon-mixture", default="0.0:0.5,0.1:0.3,0.3:0.2"
    )
    # Collection-time Monte Carlo budget from the paper protocol.  Keep the
    # larger 1,000-trial budget for the evaluation entry point only.
    parser.add_argument("--mc-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--node-index-base", type=int, default=0)
    parser.add_argument("--directed", action="store_true")
    parser.add_argument("--normalize-reward", action="store_true")
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--compact", action="store_true")
    return parser


if __name__ == "__main__":
    _collect(build_parser().parse_args())
