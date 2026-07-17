"""Exploratory data analysis for the evaluation-matrix datasets.

Produces, per dataset:
  - a stats dict (nodes, edges, features, classes, homophily, degree stats,
    sampled Gromov delta, feature sparsity, split sizes),
  - a 4-panel EDA figure (class balance, degree CCDF, hop-distance sample,
    stat summary),
and across datasets:
  - the delta-stratification table (fills PREREGISTRATION.md section 2) and
  - a delta vs homophily scatter that visualizes the H1 control design.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import homophily, to_undirected

from ..data.hyperbolicity import sampled_gromov_delta
from ..data.loader import apply_split, load_dataset
from .plotstyle import CLASS_COLORS, apply_style, class_names_for


def dataset_stats(data, name: str, delta_samples: int = 5000,
                  seed: int = 0) -> dict:
    ei = to_undirected(data.edge_index)
    n = data.num_nodes
    deg = torch.zeros(n, dtype=torch.long).index_add_(
        0, ei[0], torch.ones(ei.size(1), dtype=torch.long))
    y = data.y
    counts = torch.bincount(y).tolist()
    delta = sampled_gromov_delta(ei, n, n_samples=delta_samples, seed=seed)

    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(ei.t().tolist())
    lcc = max(nx.connected_components(g), key=len)

    stats = {
        "dataset": name,
        "nodes": n,
        "edges_undirected": ei.size(1) // 2,
        "features": data.num_features,
        "classes": len(counts),
        "class_counts": counts,
        "class_imbalance": max(counts) / max(1, min(counts)),
        "edge_homophily": float(homophily(ei, y, method="edge")),
        "mean_degree": float(deg.float().mean()),
        "max_degree": int(deg.max()),
        "feature_density": float((data.x != 0).float().mean()),
        "largest_cc_frac": len(lcc) / n,
        "delta_mean": delta["delta_mean"],
        "delta_max": delta["delta_max"],
    }
    for split in ("train_mask", "val_mask", "test_mask"):
        if hasattr(data, split) and getattr(data, split) is not None:
            m = getattr(data, split)
            if m.dim() == 1:
                stats[split.replace("_mask", "_size")] = int(m.sum())
    return stats


def eda_figure(data, name: str, stats: dict, save_path: str) -> None:
    apply_style()
    ei = to_undirected(data.edge_index)
    n = data.num_nodes
    deg = torch.zeros(n, dtype=torch.long).index_add_(
        0, ei[0], torch.ones(ei.size(1), dtype=torch.long)).numpy()
    y = data.y.numpy()
    k = stats["classes"]
    names = class_names_for(name, k)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    # (a) class balance
    ax = axes[0]
    counts = stats["class_counts"]
    ax.bar(range(k), counts, color=CLASS_COLORS[:k], alpha=0.85)
    ax.set_xticks(range(k))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Nodes")
    ax.set_title("(a) Class balance")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    # (b) degree CCDF (log-log) — scale-free check motivating hyperbolic geometry
    ax = axes[1]
    ds = np.sort(deg[deg > 0])
    ccdf = 1.0 - np.arange(len(ds)) / len(ds)
    ax.loglog(ds, ccdf, ".", ms=3, color="#1f77b4", alpha=0.7)
    ax.set_xlabel("Degree $k$")
    ax.set_ylabel("$P(K \\geq k)$")
    ax.set_title("(b) Degree CCDF (log–log)")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)

    # (c) per-class mean degree — links structure to class
    ax = axes[2]
    mean_deg = [deg[y == c].mean() if (y == c).sum() else 0 for c in range(k)]
    std_deg = [deg[y == c].std() if (y == c).sum() else 0 for c in range(k)]
    ax.bar(range(k), mean_deg, yerr=std_deg, color=CLASS_COLORS[:k],
           alpha=0.85, capsize=3)
    ax.set_xticks(range(k))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean degree ± std")
    ax.set_title("(c) Degree by class")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    # (d) stats panel
    ax = axes[3]
    ax.axis("off")
    lines = [
        f"nodes: {stats['nodes']:,}",
        f"edges: {stats['edges_undirected']:,}",
        f"features: {stats['features']:,}"
        f"  (density {stats['feature_density']:.3f})",
        f"classes: {stats['classes']}"
        f"  (imbalance {stats['class_imbalance']:.1f}x)",
        f"edge homophily: {stats['edge_homophily']:.3f}",
        f"mean degree: {stats['mean_degree']:.2f}"
        f"  (max {stats['max_degree']})",
        f"largest CC: {stats['largest_cc_frac'] * 100:.1f}%",
        f"Gromov delta (sampled): mean {stats['delta_mean']:.3f},"
        f" max {stats['delta_max']:.1f}",
    ]
    for key, label in (("train_size", "train"), ("val_size", "val"),
                       ("test_size", "test")):
        if key in stats:
            lines.append(f"{label} nodes: {stats[key]:,}")
    ax.text(0.02, 0.95, "\n".join(lines), va="top", family="monospace",
            fontsize=10, transform=ax.transAxes)
    ax.set_title("(d) Summary")

    fig.suptitle(f"EDA — {name}", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def delta_stratification_figure(all_stats: list[dict], save_path: str) -> None:
    """delta vs homophily scatter — the H1 stratification design in one plot."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    for s in all_stats:
        ax.scatter(s["delta_mean"], s["edge_homophily"],
                   s=np.sqrt(s["nodes"]) * 2, alpha=0.7, zorder=3)
        ax.annotate(s["dataset"], (s["delta_mean"], s["edge_homophily"]),
                    xytext=(6, 4), textcoords="offset points", fontsize=10)
    ax.set_xlabel("Sampled Gromov $\\delta$ (mean) — lower = more tree-like")
    ax.set_ylabel("Edge homophily")
    ax.set_title("Dataset stratification for H1\n"
                 "(marker size $\\propto \\sqrt{N}$)")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def delta_table_markdown(all_stats: list[dict]) -> str:
    """Fills PREREGISTRATION.md section 2. Roles from delta terciles."""
    rows = sorted(all_stats, key=lambda s: s["delta_mean"])
    lo = rows[0]["delta_mean"]
    hi = rows[-1]["delta_mean"]
    out = ["| Dataset | delta_mean | Role |", "|---|---|---|"]
    for s in rows:
        if s["delta_mean"] <= lo + (hi - lo) / 3:
            role = "positive control (low delta)"
        elif s["delta_mean"] >= lo + 2 * (hi - lo) / 3:
            role = "negative control (high delta)"
        else:
            role = "benchmark"
        out.append(f"| {s['dataset']} | {s['delta_mean']:.3f} | {role} |")
    return "\n".join(out)


def run_eda(datasets: list[str], out_dir: str = "experiments/figures/eda",
            root: str = "data") -> list[dict]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_stats = []
    for name in datasets:
        ds = load_dataset(name, root)
        data = ds[0].clone()
        split = "ratio" if name.lower() == "disease" else "public"
        data = apply_split(data, mode=split)
        stats = dataset_stats(data, name)
        all_stats.append(stats)
        eda_figure(data, name, stats, str(out / f"eda_{name}.pdf"))
        print(f"[EDA] {name}: N={stats['nodes']} delta={stats['delta_mean']:.3f}"
              f" homophily={stats['edge_homophily']:.3f}")
    delta_stratification_figure(all_stats, str(out / "delta_stratification.pdf"))
    with open(out / "eda_stats.json", "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)
    with open(out / "delta_table.md", "w", encoding="utf-8") as f:
        f.write(delta_table_markdown(all_stats) + "\n")
    return all_stats
