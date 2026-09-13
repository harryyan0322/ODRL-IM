"""Graph-partition-aware seed selection for MPGNN inference."""

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from paper_protocol import bidirected_edge_index


@dataclass
class PartitionInferenceResult:
    seeds: list
    partition_labels: list
    partition_sizes: list
    partition_quotas: list
    partition_quality: list
    num_partitions: int
    edge_cut: int
    partition_backend: str
    selection_mode: str
    partition_modularity: float = 0.0
    partition_resolution: float = 1.0

    def to_dict(self):
        return asdict(self)


def read_edge_index(path, node_index_base=0, directed=False):
    """Read an edge list into a contiguous PyG edge-index tensor."""
    if not directed:
        edges, num_nodes = bidirected_edge_index(path, node_index_base)
        return torch.tensor(edges, dtype=torch.long), num_nodes
    edges = []
    nodes = set()
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith(("#", "%")):
                continue
            fields = line.split()
            source = int(fields[0]) - node_index_base
            target = int(fields[1]) - node_index_base
            if source < 0 or target < 0:
                raise ValueError("node_index_base produces a negative node identifier")
            edges.append((source, target))
            nodes.update((source, target))
            if not directed:
                edges.append((target, source))
    if not nodes:
        raise ValueError(f"Graph contains no edges: {path}")
    num_nodes = max(nodes) + 1
    if len(nodes) != num_nodes:
        raise ValueError("Graph node identifiers must be consecutive from zero")
    return torch.tensor(edges, dtype=torch.long).t().contiguous(), num_nodes


def _undirected_adjacency(edge_index, num_nodes):
    adjacency = [set() for _ in range(num_nodes)]
    edges = np.asarray(edge_index.detach().cpu(), dtype=np.int64)
    if edges.shape[0] != 2:
        raise ValueError(f"edge_index must have shape (2, E), got {edges.shape}")
    for source, target in edges.T:
        source = int(source)
        target = int(target)
        if source == target:
            continue
        adjacency[source].add(target)
        adjacency[target].add(source)
    return [sorted(neighbors) for neighbors in adjacency]


def partition_graph(
    edge_index,
    num_nodes,
    num_partitions=None,
    seed=42,
    resolution=1.0,
    n_iterations=-1,
):
    """Discover natural communities on the undirected projection with Leiden."""
    try:
        import igraph as ig
        import leidenalg
    except ImportError as error:
        raise RuntimeError(
            "Community-aware inference requires python-igraph and leidenalg. "
            "Install them with `pip install python-igraph leidenalg`."
        ) from error

    del num_partitions  # Leiden determines the number of communities.
    edges = np.asarray(edge_index.detach().cpu(), dtype=np.int64)
    undirected_edges = {
        (min(int(source), int(target)), max(int(source), int(target)))
        for source, target in edges.T
        if int(source) != int(target)
    }
    graph = ig.Graph(
        n=int(num_nodes), edges=sorted(undirected_edges), directed=False
    )
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        resolution_parameter=float(resolution),
        n_iterations=int(n_iterations),
        seed=int(seed),
    )
    membership = partition.membership
    labels = np.asarray(membership, dtype=np.int64)
    if labels.shape != (num_nodes,):
        raise RuntimeError("Leiden returned an invalid community assignment")
    modularity = float(graph.modularity(labels.tolist())) if graph.ecount() else 0.0
    return labels, modularity, "leiden_rb_configuration"


def induced_subgraph(edge_index, nodes):
    """Return local node ids and the induced edge index for one partition."""
    device = edge_index.device
    nodes = torch.as_tensor(nodes, dtype=torch.long, device=device)
    if nodes.numel() == 0:
        return nodes, torch.empty((2, 0), dtype=torch.long, device=device)
    edge_max = int(edge_index.max().item()) if edge_index.numel() else -1
    num_nodes = max(edge_max, int(nodes.max().item())) + 1
    local_ids = torch.full((num_nodes,), -1, dtype=torch.long, device=device)
    local_ids[nodes] = torch.arange(nodes.numel(), device=device)
    source_local = local_ids[edge_index[0]]
    target_local = local_ids[edge_index[1]]
    mask = (source_local >= 0) & (target_local >= 0)
    local_edge_index = torch.stack((source_local[mask], target_local[mask]), dim=0)
    return nodes, local_edge_index


def _stable_topk(scores, count):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    count = min(max(int(count), 0), scores.size)
    if count == 0:
        return np.empty(0, dtype=np.int64)
    node_ids = np.arange(scores.size, dtype=np.int64)
    return np.lexsort((node_ids, -scores))[:count]


def _sequential_greedy_seeds(scores, edge_index, budget):
    """Greedy score re-ranking with a deterministic one-hop redundancy penalty.

    The actor is evaluated once. At each step, candidates are re-ranked by
    their actor score minus the overlap of their closed one-hop region with
    already selected regions. This is a decoder-level baseline, not retraining.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    budget = min(max(int(budget), 0), scores.size)
    if budget == 0:
        return []
    adjacency = [set([i]) for i in range(scores.size)]
    edges = np.asarray(edge_index.detach().cpu(), dtype=np.int64)
    for source, target in edges.T:
        source, target = int(source), int(target)
        if 0 <= source < scores.size and 0 <= target < scores.size:
            adjacency[source].add(target)
            adjacency[target].add(source)
    scale = float(np.std(scores))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    remaining = set(range(scores.size))
    selected = []
    covered = set()
    for _ in range(budget):
        best = None
        for node in remaining:
            region = adjacency[node]
            overlap = len(region & covered) / max(len(region), 1)
            value = float(scores[node]) / scale - overlap
            key = (value, float(scores[node]), -int(node))
            if best is None or key > best[0]:
                best = (key, node, region)
        _, node, region = best
        selected.append(int(node))
        remaining.remove(node)
        covered.update(region)
    return selected


def allocate_partition_quotas(partition_scores, partition_sizes, budget):
    """Give each part coverage, then allocate remaining slots by score quality."""
    partition_sizes = np.asarray(partition_sizes, dtype=np.int64)
    quality = np.asarray(
        [float(np.mean(scores)) if len(scores) else -np.inf for scores in partition_scores],
        dtype=np.float64,
    )
    active = np.flatnonzero(partition_sizes > 0)
    if budget < active.size:
        active = active[np.lexsort((active, -quality[active]))[:budget]]
    quotas = np.zeros(partition_sizes.size, dtype=np.int64)
    quotas[active] = 1
    remaining = int(budget) - int(quotas.sum())
    while remaining > 0:
        available = np.flatnonzero(quotas < partition_sizes)
        if available.size == 0:
            break
        candidates = []
        for part in available:
            ranked = _stable_topk(partition_scores[part], quotas[part] + 1)
            next_score = float(partition_scores[part][ranked[-1]])
            candidates.append((-next_score, int(part)))
        _, selected_part = min(candidates)
        quotas[selected_part] += 1
        remaining -= 1
    return quotas, quality


def select_partitioned_seeds(
    policy,
    obs,
    edge_index,
    budget,
    num_partitions=None,
    partition_seed=42,
    selection_mode="partitioned",
    partition_resolution=1.0,
    partition_iterations=-1,
):
    """Score the full graph once and decode Global or Leiden Community Top-K."""
    try:
        device = next(policy.parameters()).device
    except StopIteration:
        device = obs.device if torch.is_tensor(obs) else torch.device("cpu")
    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    edge_index = torch.as_tensor(edge_index, dtype=torch.long, device=obs.device)
    num_nodes = int(obs.shape[0])
    budget = min(max(int(budget), 1), num_nodes)
    policy.eval()
    with torch.no_grad():
        scores = policy(obs, edge_index).reshape(-1).detach().cpu().numpy()
    if selection_mode == "global_topk":
        seeds = _stable_topk(scores, budget).tolist()
        return PartitionInferenceResult(
            seeds=seeds,
            partition_labels=[0] * num_nodes,
            partition_sizes=[num_nodes],
            partition_quotas=[budget],
            partition_quality=[float(np.mean(scores))],
            num_partitions=1,
            edge_cut=0,
            partition_backend="none",
            selection_mode=selection_mode,
        )
    if selection_mode == "sequential_greedy":
        seeds = _sequential_greedy_seeds(scores, edge_index, budget)
        return PartitionInferenceResult(
            seeds=seeds,
            partition_labels=[0] * num_nodes,
            partition_sizes=[num_nodes],
            partition_quotas=[budget],
            partition_quality=[float(np.mean(scores))],
            num_partitions=1,
            edge_cut=0,
            partition_backend="one_hop_redundancy_penalty",
            selection_mode=selection_mode,
        )
    if selection_mode != "partitioned":
        raise ValueError(f"Unknown selection_mode: {selection_mode}")

    labels, modularity, backend = partition_graph(
        edge_index,
        num_nodes,
        num_partitions=num_partitions,
        seed=partition_seed,
        resolution=partition_resolution,
        n_iterations=partition_iterations,
    )
    partition_nodes = []
    partition_scores = []
    for part in range(int(labels.max()) + 1):
        global_nodes = np.flatnonzero(labels == part)
        if global_nodes.size == 0:
            continue
        partition_nodes.append(global_nodes)
        partition_scores.append(scores[global_nodes])

    partition_sizes = [len(nodes) for nodes in partition_nodes]
    quotas, quality = allocate_partition_quotas(partition_scores, partition_sizes, budget)
    seeds = []
    for nodes, scores, quota in zip(partition_nodes, partition_scores, quotas):
        seeds.extend(int(nodes[index]) for index in _stable_topk(scores, quota))
    if len(seeds) != budget or len(set(seeds)) != budget:
        raise RuntimeError("Partition-aware inference failed to return K unique nodes")

    compact_labels = np.empty(num_nodes, dtype=np.int64)
    for compact_part, nodes in enumerate(partition_nodes):
        compact_labels[nodes] = compact_part
    return PartitionInferenceResult(
        seeds=seeds,
        partition_labels=compact_labels.tolist(),
        partition_sizes=partition_sizes,
        partition_quotas=quotas.tolist(),
        partition_quality=quality.tolist(),
        num_partitions=len(partition_nodes),
        edge_cut=0,
        partition_backend=backend,
        selection_mode=selection_mode,
        partition_modularity=modularity,
        partition_resolution=float(partition_resolution),
    )
