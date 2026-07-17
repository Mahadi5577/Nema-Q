"""Sampled Gromov delta-hyperbolicity.

delta = 0 for trees; larger delta = less tree-like. Exact computation is
O(n^4), so we use the standard sampled 4-point condition on shortest-path
distances within the largest connected component. This produces the
dataset-stratification table that hypothesis H1 depends on.
"""
import networkx as nx
import numpy as np


def _four_point_delta(d, quad) -> float:
    a, b, c, x = quad
    s1 = d[a][b] + d[c][x]
    s2 = d[a][c] + d[b][x]
    s3 = d[a][x] + d[b][c]
    top2 = sorted((s1, s2, s3))[-2:]
    return (top2[1] - top2[0]) / 2.0


def sampled_gromov_delta(edge_index, num_nodes: int, n_samples: int = 5000,
                         max_component: int = 3000, seed: int = 0) -> dict:
    """Return {'delta_mean', 'delta_max', 'n_samples'} on sampled 4-tuples."""
    rng = np.random.default_rng(seed)
    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    g.add_edges_from(edge_index.t().tolist())
    comp = max(nx.connected_components(g), key=len)
    nodes = list(comp)
    if len(nodes) > max_component:
        nodes = list(rng.choice(nodes, size=max_component, replace=False))
    sub = g.subgraph(nodes)
    d = dict(nx.all_pairs_shortest_path_length(sub))

    deltas = []
    node_arr = np.array(list(sub.nodes))
    while len(deltas) < n_samples:
        quad = rng.choice(node_arr, size=4, replace=False)
        try:
            deltas.append(_four_point_delta(d, quad))
        except KeyError:
            continue  # disconnected pair within sampled subgraph
    deltas = np.array(deltas)
    return {
        "delta_mean": float(deltas.mean()),
        "delta_max": float(deltas.max()),
        "n_samples": len(deltas),
    }
