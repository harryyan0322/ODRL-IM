"""Single source of truth for the graph, WC-IC and node-feature protocol.

The processed ``soc-*`` files used by the manuscript store one row per
undirected simple edge.  The diffusion/message-passing graph is the explicit
bidirected projection of that edge list.
"""

from pathlib import Path

import numpy as np
import scipy.sparse as sp


PROTOCOL_ID = "r2_pdf_locked_v1"
FEATURE_NAMES = ("in_degree", "out_degree", "pagerank", "seed_availability")


def read_stored_undirected_edges(path, node_index_base=0):
    edges = set()
    nodes = set()
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith(("#", "%")):
                continue
            source, target = map(int, line.split()[:2])
            source -= int(node_index_base)
            target -= int(node_index_base)
            if source < 0 or target < 0:
                continue
            if source == target:
                nodes.add(source)
                continue
            edge = (min(source, target), max(source, target))
            edges.add(edge)
            nodes.update(edge)
    if not nodes:
        raise ValueError(f"Graph contains no non-self edges: {path}")
    num_nodes = max(nodes) + 1
    if nodes != set(range(num_nodes)):
        raise ValueError("Graph node identifiers must be consecutive from zero")
    return num_nodes, sorted(edges)


def bidirected_edges(path, node_index_base=0):
    num_nodes, undirected = read_stored_undirected_edges(path, node_index_base)
    directed = []
    for source, target in undirected:
        directed.extend(((source, target), (target, source)))
    return num_nodes, directed


def bidirected_edge_index(path, node_index_base=0):
    num_nodes, edges = bidirected_edges(path, node_index_base)
    return np.asarray(edges, dtype=np.int64).T, num_nodes


def pagerank(num_nodes, edges, damping=0.85, iterations=100, tolerance=1e-10):
    if not edges:
        return np.full(num_nodes, 1.0 / max(num_nodes, 1), dtype=np.float32)
    rows = np.asarray([source for source, _ in edges], dtype=np.int64)
    cols = np.asarray([target for _, target in edges], dtype=np.int64)
    out_degree = np.bincount(rows, minlength=num_nodes).astype(np.float64)
    transition = sp.csr_matrix(
        (np.ones(len(edges), dtype=np.float64) / np.maximum(out_degree[rows], 1.0),
         (cols, rows)),
        shape=(num_nodes, num_nodes),
    )
    value = np.full(num_nodes, 1.0 / num_nodes, dtype=np.float64)
    teleport = (1.0 - damping) / num_nodes
    for _ in range(int(iterations)):
        updated = damping * np.asarray(transition.dot(value)).reshape(-1) + teleport
        dangling = value[out_degree == 0].sum()
        updated += damping * dangling / num_nodes
        if np.max(np.abs(updated - value)) <= tolerance:
            value = updated
            break
        value = updated
    total = value.sum()
    return (value / total if total else np.full(num_nodes, 1.0 / num_nodes)).astype(np.float32)


def pdf_node_features(path, node_index_base=0, seed_availability=None):
    """Return the PDF's four ordered node features and bidirected edges."""
    num_nodes, edges = bidirected_edges(path, node_index_base)
    sources = np.asarray([source for source, _ in edges], dtype=np.int64)
    targets = np.asarray([target for _, target in edges], dtype=np.int64)
    out_degree = np.bincount(sources, minlength=num_nodes).astype(np.float32)
    in_degree = np.bincount(targets, minlength=num_nodes).astype(np.float32)
    rank = pagerank(num_nodes, edges)
    available = np.zeros(num_nodes, dtype=np.float32)
    if seed_availability is not None:
        available[np.asarray(seed_availability, dtype=np.int64)] = 1.0
    features = np.column_stack((in_degree, out_degree, rank, available)).astype(np.float32)
    return features, edges


def protocol_metadata():
    return {
        "protocol_id": PROTOCOL_ID,
        "graph_mode": "bidirected_from_undirected_edge_list",
        "wc_probability": "inverse_target_indegree",
        "diffusion_model": "weighted_cascade_ic",
        "node_features": list(FEATURE_NAMES),
        "node_feature_dim": len(FEATURE_NAMES),
    }
