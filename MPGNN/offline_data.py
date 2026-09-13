"""Configuration-driven loading and mixing of offline MPGNN datasets."""

import hashlib
import json
import math
import os
import pickle
from pathlib import Path

import numpy as np


LEGACY_KEYS = {
    "obs_list": "obs",
    "next_obs_list": "next_obs",
    "action_list": "actions",
    "reward_list": "rewards",
}
REQUIRED_KEYS = ("obs", "next_obs", "actions", "rewards", "adj_matrix")


class OfflineDataConfigError(ValueError):
    """Raised when an offline-data manifest is invalid."""


def _install_numpy_pickle_compat():
    """Allow NumPy 2.x pickles to load in the project's NumPy 1.x runtime."""
    import sys

    try:
        import numpy.core as numpy_core

        sys.modules.setdefault("numpy._core", numpy_core)
        sys.modules.setdefault("numpy._core.multiarray", numpy_core.multiarray)
    except (AttributeError, ImportError):
        pass


def _resolve_path(value, base_dir):
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    path = Path(expanded)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def load_json(path):
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as file:
        return json.load(file), path


def load_pickle_dataset(path):
    """Load a legacy or normalized pickle without mutating the stored object."""
    _install_numpy_pickle_compat()
    path = Path(path)
    with path.open("rb") as file:
        raw = pickle.load(file)
    if not isinstance(raw, dict):
        raise OfflineDataConfigError(f"Offline dataset must be a dict: {path}")

    if raw.get("format") == "mpgnn_compact_v1":
        initial_obs = np.asarray(raw["initial_obs"], dtype=np.float32)
        seed_sets = raw["seed_sets"]
        rewards = raw["rewards"]
        if len(seed_sets) != len(rewards):
            raise OfflineDataConfigError(f"Inconsistent compact sample counts in {path}")
        obs = []
        next_obs = []
        actions = []
        num_nodes = initial_obs.shape[0]
        for seeds in seed_sets:
            seed_indices = np.asarray(seeds, dtype=np.int64)
            state = initial_obs.copy()
            state[:, -1] = 0.0
            terminal = state.copy()
            terminal[seed_indices, -1] = 1.0
            action = np.zeros((num_nodes, 1), dtype=np.float32)
            action[seed_indices, 0] = 1.0
            obs.append(state)
            next_obs.append(terminal)
            actions.append(action)
        return {
            "obs": obs,
            "next_obs": next_obs,
            "actions": actions,
            "rewards": list(rewards),
            "adj_matrix": raw["adj_matrix"],
        }

    data = dict(raw)
    for old_key, new_key in LEGACY_KEYS.items():
        if new_key not in data and old_key in data:
            data[new_key] = data[old_key]
    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        raise OfflineDataConfigError(f"Missing keys {missing} in {path}")

    lengths = [len(data[key]) for key in ("obs", "next_obs", "actions", "rewards")]
    if len(set(lengths)) != 1:
        raise OfflineDataConfigError(f"Inconsistent sample counts in {path}: {lengths}")
    return data


def load_compact_transition_store(path):
    """Load compact data without materializing one observation per sample."""
    _install_numpy_pickle_compat()
    path = Path(path).resolve()
    with path.open("rb") as file:
        raw = pickle.load(file)
    if raw.get("format") != "mpgnn_compact_v1":
        return load_pickle_dataset(path)
    seed_sets = [
        np.asarray(seeds, dtype=np.int64) for seeds in raw["seed_sets"]
    ]
    if len(seed_sets) != len(raw["rewards"]):
        raise OfflineDataConfigError(
            f"Inconsistent compact sample counts in {path}"
        )
    return {
        "format": "mpgnn_compact_v1",
        "initial_obs": np.asarray(raw["initial_obs"], dtype=np.float32),
        "seed_sets": seed_sets,
        "rewards": np.asarray(raw["rewards"], dtype=np.float32),
        "adj_matrix": raw["adj_matrix"],
        "path": str(path),
    }


def load_pickle_shards(paths):
    """Load compatible pickle shards as one logical behavior-policy source."""
    shards = [load_pickle_dataset(path) for path in paths]
    if not shards:
        raise OfflineDataConfigError("At least one pickle shard is required")
    reference = shards[0]
    combined = {key: [] for key in ("obs", "next_obs", "actions", "rewards")}
    for shard_index, shard in enumerate(shards):
        if shard_index:
            _validate_source_compatibility(reference, shard, f"shard_{shard_index}")
        for key in combined:
            combined[key].extend(shard[key])
    combined["adj_matrix"] = reference["adj_matrix"]
    return combined


def _stable_seed(base_seed, source_name):
    digest = hashlib.sha256(f"{base_seed}:{source_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2 ** 32)


def _weighted_source_schedule(names, weights, total_samples):
    """Return a deterministic weighted schedule whose prefixes stay nested."""
    counts = np.zeros(len(names), dtype=np.int64)
    schedule = []
    for step in range(total_samples):
        desired = (step + 1) * weights
        source_index = int(np.argmax(desired - counts))
        counts[source_index] += 1
        schedule.append(names[source_index])
    return schedule, dict(zip(names, counts.tolist()))


def _sample_indices(size, count, replacement, seed, source_name):
    if size <= 0:
        raise OfflineDataConfigError(f"Source {source_name!r} contains no samples")
    if not replacement and count > size:
        raise OfflineDataConfigError(
            f"Source {source_name!r} requests {count} samples but only {size} are "
            "available. Collect more data or explicitly enable replacement."
        )

    rng = np.random.default_rng(_stable_seed(seed, source_name))
    if not replacement:
        return rng.permutation(size)[:count].tolist()

    result = []
    while len(result) < count:
        result.extend(rng.permutation(size).tolist())
    return result[:count]


def _validate_source_compatibility(reference, candidate, source_name):
    reference_obs = np.asarray(reference["obs"][0])
    candidate_obs = np.asarray(candidate["obs"][0])
    if reference_obs.shape != candidate_obs.shape:
        raise OfflineDataConfigError(
            f"Source {source_name!r} observation shape {candidate_obs.shape} does "
            f"not match {reference_obs.shape}"
        )
    reference_action = np.asarray(reference["actions"][0])
    candidate_action = np.asarray(candidate["actions"][0])
    if reference_action.shape != candidate_action.shape:
        raise OfflineDataConfigError(
            f"Source {source_name!r} action shape {candidate_action.shape} does "
            f"not match {reference_action.shape}"
        )
    if np.asarray(reference["adj_matrix"]).shape != np.asarray(candidate["adj_matrix"]).shape:
        raise OfflineDataConfigError(f"Source {source_name!r} uses a different graph size")


def _normalize_manifest(config, config_path):
    if config.get("version") != 1:
        raise OfflineDataConfigError("Only offline-data manifest version 1 is supported")
    total_samples = int(config.get("total_samples", 0))
    if total_samples <= 0:
        raise OfflineDataConfigError("total_samples must be positive")

    raw_sources = config.get("sources", [])
    if not raw_sources:
        raise OfflineDataConfigError("At least one data source is required")

    base_dir = config_path.parent
    sources = []
    names = set()
    for source in raw_sources:
        name = str(source.get("name", "")).strip()
        if not name or name in names:
            raise OfflineDataConfigError(f"Invalid or duplicate source name: {name!r}")
        names.add(name)
        weight = float(source.get("weight", 0.0))
        if weight <= 0:
            raise OfflineDataConfigError(f"Source {name!r} must have a positive weight")
        raw_paths = source.get("paths")
        if raw_paths is None and source.get("path") is not None:
            raw_paths = [source["path"]]
        if not raw_paths:
            raise OfflineDataConfigError(f"Source {name!r} is missing path or paths")
        sources.append({
            "name": name,
            "paths": [_resolve_path(path, base_dir) for path in raw_paths],
            "weight": weight,
            "replacement": bool(source.get("replacement", False)),
        })

    weight_sum = sum(source["weight"] for source in sources)
    for source in sources:
        source["weight"] /= weight_sum
    return total_samples, sources


def load_offline_dataset(config_path):
    """Materialize the mixed dataset described by a JSON manifest."""
    config, config_path = load_json(config_path)
    total_samples, sources = _normalize_manifest(config, config_path)
    seed = int(config.get("sampling_seed", 0))

    names = [source["name"] for source in sources]
    weights = np.asarray([source["weight"] for source in sources], dtype=np.float64)
    schedule, source_counts = _weighted_source_schedule(names, weights, total_samples)

    loaded = {}
    source_indices = {}
    reference = None
    for source in sources:
        name = source["name"]
        dataset = load_pickle_shards(source["paths"])
        if reference is None:
            reference = dataset
        else:
            _validate_source_compatibility(reference, dataset, name)
        loaded[name] = dataset
        source_indices[name] = _sample_indices(
            len(dataset["rewards"]),
            source_counts[name],
            source["replacement"],
            seed,
            name,
        )

    cursors = {name: 0 for name in names}
    mixed = {"obs": [], "next_obs": [], "actions": [], "rewards": []}
    labels = []
    original_indices = []
    for name in schedule:
        cursor = cursors[name]
        original_index = source_indices[name][cursor]
        cursors[name] += 1
        dataset = loaded[name]
        for key in mixed:
            mixed[key].append(dataset[key][original_index])
        labels.append(name)
        original_indices.append(original_index)

    mixed["adj_matrix"] = reference["adj_matrix"]
    terminal_value = bool(config.get("terminal", True))
    mixed["terminals"] = np.full(total_samples, terminal_value, dtype=np.float32)
    mixed["source_labels"] = labels
    mixed["source_indices"] = np.asarray(original_indices, dtype=np.int64)
    mixed["manifest"] = {
        "path": str(config_path),
        "name": config.get("name", config_path.stem),
        "dataset": config.get("dataset"),
        "graph_path": (
            str(_resolve_path(config["graph_path"], config_path.parent))
            if config.get("graph_path")
            else None
        ),
        "budget_percent": config.get("budget_percent"),
        "sampling_seed": seed,
        "source_counts": source_counts,
        "source_weights": dict(zip(names, weights.tolist())),
    }
    return mixed


def _topk_seed_set(action, budget_nodes):
    scores = np.asarray(action).reshape(-1)
    budget_nodes = min(max(int(budget_nodes), 1), scores.size)
    indices = np.argpartition(scores, -budget_nodes)[-budget_nodes:]
    return frozenset(int(index) for index in indices)


def dataset_statistics(data, budget_percent=None, max_pairs=10000, seed=0):
    """Compute source, reward, and action-coverage diagnostics."""
    rewards = np.asarray(data["rewards"], dtype=np.float64)
    num_samples = len(rewards)
    if num_samples == 0:
        raise OfflineDataConfigError("Cannot inspect an empty dataset")

    labels = data.get("source_labels", ["unknown"] * num_samples)
    source_counts = {}
    for label in labels:
        source_counts[label] = source_counts.get(label, 0) + 1

    num_nodes = np.asarray(data["actions"][0]).reshape(-1).size
    if budget_percent is None:
        budget_percent = data.get("manifest", {}).get("budget_percent")
    if budget_percent is None:
        raise OfflineDataConfigError("budget_percent is required for coverage statistics")
    budget_nodes = max(1, int(float(budget_percent) * num_nodes / 100.0))

    seed_sets = [_topk_seed_set(action, budget_nodes) for action in data["actions"]]
    selection_counts = np.zeros(num_nodes, dtype=np.int64)
    for seed_set in seed_sets:
        selection_counts[list(seed_set)] += 1
    selected = selection_counts[selection_counts > 0]
    probabilities = selected / selected.sum()
    entropy = -np.sum(probabilities * np.log(probabilities))
    normalized_entropy = entropy / math.log(num_nodes) if num_nodes > 1 else 0.0

    pair_total = num_samples * (num_samples - 1) // 2
    pair_count = min(int(max_pairs), pair_total)
    jaccard_values = []
    if pair_count > 0:
        rng = np.random.default_rng(seed)
        if pair_total <= max_pairs:
            pairs = ((i, j) for i in range(num_samples) for j in range(i + 1, num_samples))
        else:
            sampled_pairs = set()
            while len(sampled_pairs) < pair_count:
                i, j = rng.integers(0, num_samples, size=2)
                if i != j:
                    sampled_pairs.add(tuple(sorted((int(i), int(j)))))
            pairs = iter(sampled_pairs)
        for i, j in pairs:
            intersection = len(seed_sets[i] & seed_sets[j])
            union = len(seed_sets[i] | seed_sets[j])
            jaccard_values.append(1.0 - intersection / union)

    return {
        "num_samples": num_samples,
        "num_nodes": num_nodes,
        "budget_nodes": budget_nodes,
        "source_counts": source_counts,
        "source_ratios": {name: count / num_samples for name, count in source_counts.items()},
        "reward": {
            "mean": float(rewards.mean()),
            "std": float(rewards.std()),
            "min": float(rewards.min()),
            "q10": float(np.quantile(rewards, 0.10)),
            "median": float(np.median(rewards)),
            "q90": float(np.quantile(rewards, 0.90)),
            "max": float(rewards.max()),
        },
        "coverage": {
            "selected_nodes": int(np.count_nonzero(selection_counts)),
            "node_coverage": float(np.count_nonzero(selection_counts) / num_nodes),
            "selection_entropy_normalized": float(normalized_entropy),
            "unique_seed_set_ratio": float(len(set(seed_sets)) / num_samples),
            "mean_pairwise_jaccard_distance": (
                float(np.mean(jaccard_values)) if jaccard_values else 0.0
            ),
            "jaccard_pairs": len(jaccard_values),
        },
    }


def expand_suite(suite_path, output_dir):
    """Expand an experiment suite into individual training manifests."""
    suite, suite_path = load_json(suite_path)
    if suite.get("version") != 1:
        raise OfflineDataConfigError("Only suite version 1 is supported")
    source_catalog = suite.get("sources", {})
    if not source_catalog:
        raise OfflineDataConfigError("Suite contains no source catalog")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suite_base_dir = suite_path.parent
    written = []
    for experiment in suite.get("experiments", []):
        experiment_name = experiment["name"]
        weights = experiment["weights"]
        for source_name in weights:
            if source_name not in source_catalog:
                raise OfflineDataConfigError(
                    f"Experiment {experiment_name!r} references unknown source {source_name!r}"
                )
        sample_sizes = experiment.get("total_samples", [])
        if isinstance(sample_sizes, int):
            sample_sizes = [sample_sizes]
        for total_samples in sample_sizes:
            manifest_name = f"{experiment_name}_n{int(total_samples)}"
            manifest = {
                "version": 1,
                "name": manifest_name,
                "dataset": suite.get("dataset"),
                "graph_path": (
                    str(_resolve_path(suite["graph_path"], suite_base_dir))
                    if suite.get("graph_path")
                    else None
                ),
                "budget_percent": suite.get("budget_percent"),
                "total_samples": int(total_samples),
                "sampling_seed": int(suite.get("sampling_seed", 0)),
                "terminal": bool(suite.get("terminal", True)),
                "sources": [],
            }
            for source_name, weight in weights.items():
                source_config = source_catalog[source_name]
                source_manifest = {
                    "name": source_name,
                    "weight": float(weight),
                    "replacement": bool(source_config.get("replacement", False)),
                }
                if source_config.get("paths") is not None:
                    source_manifest["paths"] = [
                        str(_resolve_path(path, suite_base_dir))
                        for path in source_config["paths"]
                    ]
                else:
                    source_manifest["path"] = str(
                        _resolve_path(source_config["path"], suite_base_dir)
                    )
                manifest["sources"].append(source_manifest)
            output_path = output_dir / f"{manifest_name}.json"
            with output_path.open("w", encoding="utf-8") as file:
                json.dump(manifest, file, indent=2, ensure_ascii=True)
                file.write("\n")
            written.append(output_path)
    return written


def save_legacy_pickle(data, output_path):
    """Save normalized in-memory data in the legacy format used by MPGNN."""
    output = {
        "obs_list": data["obs"],
        "next_obs_list": data["next_obs"],
        "action_list": data["actions"],
        "reward_list": data["rewards"],
        "adj_matrix": data["adj_matrix"],
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump(output, file, protocol=pickle.HIGHEST_PROTOCOL)


def save_compact_pickle(initial_obs, seed_sets, rewards, adj_matrix, output_path):
    """Save a compact source shard with graph state stored only once."""
    output = {
        "format": "mpgnn_compact_v1",
        "initial_obs": np.asarray(initial_obs, dtype=np.float32),
        "seed_sets": [np.asarray(seeds, dtype=np.int32) for seeds in seed_sets],
        "rewards": [float(reward) for reward in rewards],
        "adj_matrix": adj_matrix,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump(output, file, protocol=pickle.HIGHEST_PROTOCOL)
