"""Evaluate MPGNN checkpoints with partition-aware seed inference."""

import argparse
import io
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

from influence_evaluator import (
    read_weighted_cascade_graph,
    seed_dispersion_metrics,
    simulate_ic_spreads,
)
from offline_data import (
    load_compact_transition_store,
    load_offline_dataset,
    load_pickle_dataset,
)
from partition_inference import read_edge_index, select_partitioned_seeds


class _CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda value: torch.load(io.BytesIO(value), map_location="cpu")
        return super().find_class(module, name)


def _load_checkpoint(path):
    from Model import GCN, GNNQFunction  # noqa: F401 - required by old pickles

    sys.modules.setdefault("MPGNN.Model", sys.modules["Model"])
    with Path(path).open("rb") as file:
        checkpoint = _CPUUnpickler(file).load()
    if not isinstance(checkpoint, dict) or "policy" not in checkpoint:
        raise ValueError("Checkpoint must be a dictionary containing a policy")
    return checkpoint


def _load_observation(args):
    if args.community_data_config:
        manifest = json.loads(
            Path(args.community_data_config).read_text(encoding="utf-8")
        )
        communities = manifest.get("communities", [])
        if len(communities) != 1 or communities[0].get("mapping_path"):
            raise ValueError(
                "evaluate.py only accepts a one-unit full-graph community "
                "manifest; use evaluate_community_checkpoint.py for shards"
            )
        sources = communities[0]["sources"]
        paths = sources.get(args.observation_source) or next(iter(sources.values()))
        path = paths[0] if isinstance(paths, list) else paths
        store = load_compact_transition_store(path)
        obs = np.asarray(store["initial_obs"], dtype=np.float32).copy()
        obs[:, -1] = 0.0
        return obs
    if args.data_config:
        dataset = load_offline_dataset(args.data_config)
    elif args.data_path:
        dataset = load_pickle_dataset(args.data_path)
    else:
        raise ValueError("One of --data-config or --data-path is required")
    obs = np.asarray(dataset["obs"][args.observation_index], dtype=np.float32).copy()
    obs[:, -1] = 0.0
    return obs


def _representation_diagnostics(policy, obs, edge_index, max_nodes=2048):
    """Summarize hidden-state collapse without quadratic all-node storage."""
    if not hasattr(policy, "forward_with_hidden"):
        return {"available": False, "reason": "legacy_policy"}
    with torch.no_grad():
        _, hidden_states = policy.forward_with_hidden(obs, edge_index)
    diagnostics = []
    for layer_index, hidden in enumerate(hidden_states, start=1):
        flat = hidden.reshape(-1, hidden.shape[-1]).float()
        node_count = flat.shape[0]
        if node_count > max_nodes:
            indices = torch.linspace(
                0, node_count - 1, steps=max_nodes, device=flat.device
            ).long()
            flat = flat.index_select(0, indices)
        centered = flat - flat.mean(dim=0, keepdim=True)
        normalized = torch.nn.functional.normalize(flat, dim=1, eps=1e-12)
        mean_direction = normalized.mean(dim=0)
        pairwise_cosine = (
            normalized.shape[0] * mean_direction.pow(2).sum() - 1.0
        ) / max(normalized.shape[0] - 1, 1)
        diagnostics.append({
            "layer": layer_index,
            "feature_variance": float(centered.pow(2).mean().item()),
            "mean_pairwise_cosine": float(pairwise_cosine.item()),
            "sampled_nodes": int(normalized.shape[0]),
        })
    return {"available": True, "layers": diagnostics}


def evaluate(args):
    requested_device = args.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    checkpoint = _load_checkpoint(args.model_path)
    policy = checkpoint["policy"].to(device)
    policy.eval()
    obs = torch.tensor(_load_observation(args), dtype=torch.float32, device=device)
    edge_index, num_nodes = read_edge_index(
        args.graph_path,
        node_index_base=args.node_index_base,
        directed=False,
    )
    if num_nodes != obs.shape[0]:
        raise ValueError(
            f"Graph has {num_nodes} nodes but observation has {obs.shape[0]} rows"
        )
    edge_index = edge_index.to(device)
    budget_nodes = max(1, int(args.budget_percent * num_nodes / 100.0))

    started = time.perf_counter()
    inference = select_partitioned_seeds(
        policy,
        obs,
        edge_index,
        budget_nodes,
        num_partitions=args.num_partitions,
        partition_seed=args.partition_seed,
        selection_mode=args.selection_mode,
        partition_resolution=args.leiden_resolution,
        partition_iterations=args.leiden_iterations,
    )
    inference_seconds = time.perf_counter() - started
    representation_diagnostics = _representation_diagnostics(
        policy, obs, edge_index
    )

    graph = read_weighted_cascade_graph(
        args.graph_path,
        node_index_base=args.node_index_base,
        directed=args.directed,
    )
    reward_rng = np.random.default_rng(args.mc_seed)
    trial_spreads = simulate_ic_spreads(
        graph, inference.seeds, args.mc_trials, reward_rng
    )
    dispersion = seed_dispersion_metrics(graph, inference.seeds)
    spread = float(trial_spreads.mean())
    trial_std = float(trial_spreads.std(ddof=1)) if args.mc_trials > 1 else 0.0
    trial_coverage_rates = 100.0 * trial_spreads / num_nodes
    coverage_rate = float(trial_coverage_rates.mean())
    coverage_rate_std = (
        float(trial_coverage_rates.std(ddof=1)) if args.mc_trials > 1 else 0.0
    )
    if args.mc_trials > 1:
        from scipy.stats import t
        critical = t.ppf(0.975, args.mc_trials - 1)
        half_width = float(critical * trial_std / np.sqrt(args.mc_trials))
        coverage_half_width = float(
            critical * coverage_rate_std / np.sqrt(args.mc_trials)
        )
    else:
        half_width = 0.0
        coverage_half_width = 0.0
    result = inference.to_dict()
    result.update({
        "model_path": str(Path(args.model_path).resolve()),
        "data_config": str(Path(args.data_config).resolve()) if args.data_config else None,
        "community_data_config": (
            str(Path(args.community_data_config).resolve())
            if args.community_data_config else None
        ),
        "data_path": str(Path(args.data_path).resolve()) if args.data_path else None,
        "graph_path": str(Path(args.graph_path).resolve()),
        "num_nodes": num_nodes,
        "budget_percent": args.budget_percent,
        "budget_nodes": budget_nodes,
        "diffusion_model": "weighted_cascade_ic",
        "mc_trials": args.mc_trials,
        "mc_seed": args.mc_seed,
        "requested_device": args.device,
        "effective_device": str(device),
        "training_seed": args.training_seed,
        "expected_spread": spread,
        "spread_std": trial_std,
        "spread_ci95_low": spread - half_width,
        "spread_ci95_high": spread + half_width,
        "mc_trial_spreads": trial_spreads.tolist(),
        "normalized_spread": float(spread / num_nodes),
        "coverage_rate_percent": coverage_rate,
        "coverage_rate_std_mc_percent": coverage_rate_std,
        "coverage_rate_ci95_low_mc_percent": coverage_rate - coverage_half_width,
        "coverage_rate_ci95_high_mc_percent": coverage_rate + coverage_half_width,
        "coverage_rate_formula": "100 * activated_nodes / total_nodes",
        "coverage_rate_includes_seed_nodes": True,
        "inference_seconds": inference_seconds,
        "representation_diagnostics": representation_diagnostics,
        "seed_dispersion": dispersion,
    })

    output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")
        print(f"Saved evaluation to {output_path.resolve()}")
    else:
        print(output, end="")
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--data-config")
    source_group.add_argument("--data-path")
    source_group.add_argument("--community-data-config")
    parser.add_argument("--observation-source", default="s2v")
    parser.add_argument("--graph-path", required=True)
    parser.add_argument("--budget-percent", type=float, required=True)
    parser.add_argument("--output")
    parser.add_argument("--selection-mode", choices=("partitioned", "global_topk", "sequential_greedy"), default="partitioned")
    parser.add_argument("--num-partitions", type=int, default=None)
    parser.add_argument("--partition-seed", type=int, default=42)
    parser.add_argument("--leiden-resolution", type=float, default=1.0)
    parser.add_argument("--leiden-iterations", type=int, default=-1)
    parser.add_argument("--mc-trials", type=int, default=1000)
    parser.add_argument("--mc-seed", type=int, default=42)
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--observation-index", type=int, default=0)
    parser.add_argument("--node-index-base", type=int, default=0)
    parser.add_argument("--directed", action="store_true")
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    return parser


def main():
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()
