"""Influence-spread evaluation without deep-learning dependencies."""

from collections import deque

import numpy as np

from paper_protocol import bidirected_edges


class WeightedCascadeGraph:
    """Directed graph using p(u, v) = 1 / in_degree(v)."""

    def __init__(self, children, probabilities, num_nodes, num_edges):
        self.children = children
        self.probabilities = probabilities
        self.num_nodes = num_nodes
        self.num_edges = num_edges


def read_weighted_cascade_graph(path, node_index_base=0, directed=False):
    # The manuscript protocol stores undirected edges once and expands every
    # edge into two opposite diffusion edges. ``directed=True`` is retained as
    # a legacy escape hatch only for non-formal callers; formal code must use
    # the default protocol path.
    if directed:
        edges = []
        nodes = set()
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith(("#", "%")):
                    continue
                source, target = map(int, line.split()[:2])
                source -= node_index_base
                target -= node_index_base
                edges.append((source, target))
                nodes.update((source, target))
        if not nodes:
            raise ValueError(f"Graph contains no edges: {path}")
        num_nodes = max(nodes) + 1
        if len(nodes) != num_nodes:
            raise ValueError("Graph node identifiers must be consecutive from zero")
    else:
        num_nodes, edges = bidirected_edges(path, node_index_base)

    in_degree = np.zeros(num_nodes, dtype=np.int64)
    for _, target in edges:
        in_degree[target] += 1

    children = [[] for _ in range(num_nodes)]
    probabilities = [[] for _ in range(num_nodes)]
    for source, target in edges:
        children[source].append(target)
        probabilities[source].append(1.0 / in_degree[target])
    return WeightedCascadeGraph(children, probabilities, num_nodes, len(edges))


def simulate_ic_spreads(graph, seeds, num_trials, rng):
    """Return activated-node counts from independent IC simulations."""
    seeds = np.unique(np.asarray(seeds, dtype=np.int64).reshape(-1))
    if seeds.size == 0:
        return np.zeros(int(num_trials), dtype=np.float64)
    if seeds.min() < 0 or seeds.max() >= graph.num_nodes:
        raise ValueError("Seed identifier is outside the graph")

    spreads = np.empty(int(num_trials), dtype=np.float64)
    active = np.zeros(graph.num_nodes, dtype=np.bool_)
    for trial in range(int(num_trials)):
        active.fill(False)
        active[seeds] = True
        queue = deque(int(seed) for seed in seeds)
        activated_count = int(seeds.size)
        while queue:
            source = queue.popleft()
            targets = graph.children[source]
            if not targets:
                continue
            draws = rng.random(len(targets))
            for target, probability, draw in zip(
                targets, graph.probabilities[source], draws
            ):
                if not active[target] and draw <= probability:
                    active[target] = True
                    activated_count += 1
                    queue.append(target)
        spreads[trial] = activated_count
    return spreads


def compute_mc_spread(graph, seeds, num_trials, rng):
    """Estimate expected activated-node count under the IC model."""
    return float(simulate_ic_spreads(graph, seeds, num_trials, rng).mean())


def topk_seed_set(action, budget_nodes):
    scores = np.asarray(action).reshape(-1)
    budget_nodes = min(max(int(budget_nodes), 1), scores.size)
    return np.argpartition(scores, -budget_nodes)[-budget_nodes:]


def seed_set_action(seeds, num_nodes):
    """Encode a seed set as a source-independent K-hot action vector."""
    action = np.zeros((int(num_nodes), 1), dtype=np.float32)
    action[np.asarray(seeds, dtype=np.int64), 0] = 1.0
    return action


def seed_dispersion_metrics(graph, seeds, max_distance_sources=32):
    """Return deterministic redundancy diagnostics for one selected seed set.

    Average distance is measured on the undirected topology projection.  For
    large budgets, evenly spaced seed identifiers are used as BFS sources and
    all selected seeds remain distance targets.  Local influence overlap uses
    closed one-hop outgoing neighborhoods and is zero when those regions are
    disjoint.
    """
    seeds = np.unique(np.asarray(seeds, dtype=np.int64).reshape(-1))
    if seeds.size < 2:
        return {
            "average_seed_distance": 0.0,
            "distance_source_count": int(seeds.size),
            "distance_pair_count": 0,
            "distance_reachable_pair_ratio": 1.0,
            "local_influence_region_overlap_rate": 0.0,
            "local_influence_region_definition": "closed_out_neighborhood_1hop",
        }

    undirected = [set(children) for children in graph.children]
    for source, children in enumerate(graph.children):
        for target in children:
            undirected[target].add(source)

    source_count = min(int(max_distance_sources), int(seeds.size))
    source_positions = np.linspace(
        0, seeds.size - 1, num=source_count, dtype=np.int64
    )
    bfs_sources = seeds[source_positions]
    seed_mask = np.zeros(graph.num_nodes, dtype=np.bool_)
    seed_mask[seeds] = True
    distances = []
    possible_pairs = 0
    for source in bfs_sources:
        visited = np.full(graph.num_nodes, -1, dtype=np.int32)
        visited[int(source)] = 0
        queue = deque([int(source)])
        while queue:
            node = queue.popleft()
            next_distance = int(visited[node]) + 1
            for neighbor in undirected[node]:
                if visited[neighbor] < 0:
                    visited[neighbor] = next_distance
                    queue.append(neighbor)
        target_distances = visited[seeds]
        target_distances = target_distances[
            (seeds != source) & (target_distances >= 0)
        ]
        distances.extend(target_distances.tolist())
        possible_pairs += int(seeds.size - 1)

    region_memberships = 0
    region_union = set()
    for seed in seeds:
        region = set(graph.children[int(seed)])
        region.add(int(seed))
        region_memberships += len(region)
        region_union.update(region)
    overlap = (
        1.0 - len(region_union) / region_memberships
        if region_memberships else 0.0
    )
    return {
        "average_seed_distance": (
            float(np.mean(distances)) if distances else 0.0
        ),
        "distance_source_count": source_count,
        "distance_pair_count": len(distances),
        "distance_reachable_pair_ratio": (
            float(len(distances) / possible_pairs) if possible_pairs else 1.0
        ),
        "local_influence_region_overlap_rate": float(overlap),
        "local_influence_region_definition": "closed_out_neighborhood_1hop",
    }
