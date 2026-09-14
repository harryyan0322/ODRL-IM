"""Collect one-shot offline data from pretrained IM behavior policies."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

from influence_evaluator import (
    compute_mc_spread,
    read_weighted_cascade_graph,
    seed_set_action,
)
from paper_protocol import pdf_node_features
from offline_data import load_pickle_dataset, save_compact_pickle, save_legacy_pickle
from behavior_policy_sources import BEHAVIOR_POLICY_SOURCES


def _parse_epsilon_mixture(value):
    mixture = []
    for item in value.split(","):
        epsilon, weight = item.split(":")
        epsilon = float(epsilon)
        weight = float(weight)
        if not 0.0 <= epsilon <= 1.0 or weight <= 0:
            raise ValueError(f"Invalid epsilon mixture item: {item}")
        mixture.append((epsilon, weight))
    total = sum(weight for _, weight in mixture)
    return [(epsilon, weight / total) for epsilon, weight in mixture]


def _weighted_schedule(values, weights, size):
    counts = np.zeros(len(values), dtype=np.int64)
    schedule = []
    weights = np.asarray(weights, dtype=np.float64)
    for step in range(size):
        index = int(np.argmax((step + 1) * weights - counts))
        counts[index] += 1
        schedule.append(values[index])
    return schedule, counts.tolist()


def _agent_args(args, device):
    return SimpleNamespace(
        model=args.model,
        n_step=1,
        test=True,
        T=3,
        memory_size=1,
        bs=1,
        double_dqn=True,
        device=device,
        reg_hidden=32,
        embed_dim=50 if args.model == "Tripling" else 64,
        lr=0.001,
        model_file=str(Path(args.model_file).resolve()),
        data_loader_workers=args.data_loader_workers,
    )


def _collect(args):
    baseline_dir = Path(args.baseline_dir).resolve()
    sys.path.insert(0, str(baseline_dir))

    import torch
    import rl_agents
    import utils.graph_utils as behavior_graph_utils

    # Tripling/S2V inference on CPU can oversubscribe the host with the
    # default OpenMP pool.  A single torch worker is both reproducible and
    # substantially faster for the small batched graph-inference calls used
    # by this collector.  The value is explicit in metadata and can be
    # overridden for benchmarking.
    if args.torch_threads is not None:
        torch.set_num_threads(max(1, int(args.torch_threads)))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    requested_device = str(args.device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    if device.type == "cpu":
        original_torch_load = torch.load

        def load_cpu(*load_args, **load_kwargs):
            load_kwargs.setdefault("map_location", torch.device("cpu"))
            return original_torch_load(*load_args, **load_kwargs)

        torch.load = load_cpu

    graph = behavior_graph_utils.read_graph(
        args.graph_path,
        ind=args.node_index_base,
        directed=False,
    )
    evaluator_graph = read_weighted_cascade_graph(
        args.graph_path,
        node_index_base=args.node_index_base,
        directed=False,
    )
    template = load_pickle_dataset(args.template_data)
    num_nodes = np.asarray(template["obs"][0]).shape[0]
    if graph.num_nodes != num_nodes or evaluator_graph.num_nodes != num_nodes:
        raise ValueError("Graph and template node counts do not match")

    agent = rl_agents.DQAgent(_agent_args(args, device))
    if args.model == "Tripling":
        embedding = np.asarray(template["obs"][0], dtype=np.float32)[:, :-1]
        # The native Tripling policy still computes its own DeepWalk input;
        # formal logged observations are replaced below by the four PDF
        # features, so a legacy 101-D template is optional.
        if embedding.shape[1] in (100, 101):
            agent.graph_node_embed[id(graph)] = torch.tensor(embedding, dtype=torch.float32)

    pdf_features, pdf_edges = pdf_node_features(args.graph_path, args.node_index_base)

    epsilon_mixture = _parse_epsilon_mixture(args.epsilon_mixture)
    epsilons, epsilon_counts = _weighted_schedule(
        [item[0] for item in epsilon_mixture],
        [item[1] for item in epsilon_mixture],
        args.num_samples,
    )
    rng = np.random.default_rng(args.seed)
    budget_nodes = max(1, int(args.budget_percent * num_nodes / 100.0))
    result = {"obs": [], "next_obs": [], "actions": [], "rewards": []}
    selected_sets = []

    for batch_start in range(0, args.num_samples, args.inference_batch_size):
        batch_epsilons = epsilons[
            batch_start:batch_start + args.inference_batch_size
        ]
        batch_size = len(batch_epsilons)
        states = torch.zeros((batch_size, num_nodes), dtype=torch.long)
        batch_selected = [[] for _ in range(batch_size)]
        for _ in range(budget_nodes):
            graph_input = agent.setup_graph_input(
                [graph] * batch_size, states
            )
            with torch.no_grad():
                q_values = agent.model(graph_input).reshape(batch_size, num_nodes).detach().cpu()
            q_values[states.bool()] = -1e10
            for batch_index, epsilon in enumerate(batch_epsilons):
                available = torch.nonzero(
                    states[batch_index] == 0, as_tuple=False
                ).reshape(-1).numpy()
                if rng.random() < epsilon:
                    node = int(rng.choice(available))
                else:
                    node = int(torch.argmax(q_values[batch_index]).item())
                batch_selected[batch_index].append(node)
                states[batch_index, node] = 1

        for batch_index, selected in enumerate(batch_selected):
            sample_index = batch_start + batch_index
            action = seed_set_action(selected, num_nodes)
            template_index = sample_index % len(template["obs"])
            obs = pdf_features.copy()
            obs[:, -1] = 0.0
            next_obs = obs.copy()
            next_obs[selected, -1] = 1.0
            reward_rng = np.random.default_rng(
                args.seed * 1000003 + sample_index
            )
            reward = compute_mc_spread(
                evaluator_graph, selected, args.mc_trials, reward_rng
            )
            if args.normalize_reward:
                reward /= num_nodes
            result["obs"].append(obs)
            result["next_obs"].append(next_obs)
            result["actions"].append(action)
            result["rewards"].append(float(reward))
            selected_sets.append(selected)
            if args.progress_every and (sample_index + 1) % args.progress_every == 0:
                print(f"Collected {sample_index + 1}/{args.num_samples}")

    result["adj_matrix"] = sp.csr_matrix(
        (np.ones(len(pdf_edges), dtype=np.float32),
         ([edge[0] for edge in pdf_edges], [edge[1] for edge in pdf_edges])),
        shape=(num_nodes, num_nodes),
    )
    if args.compact:
        save_compact_pickle(
            result["obs"][0],
            selected_sets,
            result["rewards"],
            result["adj_matrix"],
            args.output,
        )
    else:
        save_legacy_pickle(result, args.output)
    metadata = {
        "source": args.source_name,
        "model": args.model,
        "upstream_repository": BEHAVIOR_POLICY_SOURCES[args.model]["repository"],
        "model_file": str(Path(args.model_file).resolve()),
        "template_data": str(Path(args.template_data).resolve()),
        "graph_path": str(Path(args.graph_path).resolve()),
        "num_samples": args.num_samples,
        "budget_percent": args.budget_percent,
        "budget_nodes": budget_nodes,
        "epsilon_mixture": [
            {"epsilon": epsilon, "weight": weight, "count": count}
            for (epsilon, weight), count in zip(epsilon_mixture, epsilon_counts)
        ],
        "diffusion_model": "weighted_cascade_ic",
        "mc_trials": args.mc_trials,
        "seed": args.seed,
        "action_encoding": "topk_k_hot",
        "directed": False,
        "protocol_id": "r2_pdf_locked_v1",
        "graph_mode": "bidirected_from_undirected_edge_list",
        "node_feature_dim": 4,
        "node_index_base": args.node_index_base,
        "device": str(device),
        "requested_device": args.device,
        "normalize_reward": args.normalize_reward,
        "storage_format": "mpgnn_compact_v1" if args.compact else "legacy",
        "inference_batch_size": args.inference_batch_size,
        "torch_threads": args.torch_threads,
    }
    metadata_path = Path(args.output).with_suffix(Path(args.output).suffix + ".json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Saved behavior source to {Path(args.output).resolve()}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--model", choices=("Tripling", "S2V_DQN"), required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--template-data", required=True)
    parser.add_argument("--graph-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-samples", type=int, required=True)
    parser.add_argument("--budget-percent", type=float, required=True)
    parser.add_argument(
        "--epsilon-mixture",
        default="0.0:0.5,0.1:0.3,0.3:0.2",
        help="comma-separated epsilon:weight pairs",
    )
    # The manuscript protocol uses 100 IC rollouts while collecting each
    # offline transition.  Formal evaluation is separate and uses 1,000.
    parser.add_argument("--mc-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--node-index-base", type=int, default=0)
    parser.add_argument("--directed", action="store_true")
    parser.add_argument("--normalize-reward", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--data-loader-workers", type=int, default=0)
    parser.add_argument("--inference-batch-size", type=int, default=32)
    parser.add_argument(
        "--torch-threads", type=int, default=1,
        help="PyTorch intra-op CPU threads for behavior inference (default: 1)",
    )
    parser.add_argument("--compact", action="store_true")
    return parser


if __name__ == "__main__":
    _collect(build_parser().parse_args())
