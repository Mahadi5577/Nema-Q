"""Produce the dataset delta-hyperbolicity table (paper: Datasets section; H1 stratification).

Usage:
    python scripts/compute_hyperbolicity.py --datasets cora citeseer pubmed texas
"""
import argparse
import json

from nemaq.data.hyperbolicity import sampled_gromov_delta
from nemaq.data.loader import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["cora", "citeseer", "pubmed"])
    ap.add_argument("--samples", type=int, default=5000)
    ap.add_argument("--out", default="experiments/hyperbolicity.json")
    args = ap.parse_args()

    table = {}
    for name in args.datasets:
        data = load_dataset(name)[0]
        stats = sampled_gromov_delta(data.edge_index, data.num_nodes,
                                     n_samples=args.samples)
        stats["num_nodes"] = data.num_nodes
        stats["num_edges"] = data.num_edges
        table[name] = stats
        print(f"{name}: delta_mean={stats['delta_mean']:.3f} "
              f"delta_max={stats['delta_max']:.1f}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)
    print(f"written -> {args.out}")


if __name__ == "__main__":
    main()
