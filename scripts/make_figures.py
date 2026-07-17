"""Build all results figures from run manifests (design rule #1: figures
consume manifests only).

Usage:
    python scripts/make_figures.py --runs experiments/runs \
        --datasets cora citeseer pubmed disease
The geometry x delta figure needs experiments/figures/eda/eda_stats.json
(run scripts/run_eda.py first); it is skipped otherwise.
"""
import argparse
import json
from pathlib import Path

from nemaq.analysis.figures import geometry_delta_plot, make_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="experiments/runs")
    ap.add_argument("--datasets", nargs="+", default=["cora"])
    ap.add_argument("--out", default="experiments/figures/results")
    ap.add_argument("--eda-stats", default="experiments/figures/eda/eda_stats.json")
    args = ap.parse_args()

    for ds in args.datasets:
        split = "ratio" if ds == "disease" else "public"
        try:
            make_all(args.runs, ds, args.out, split=split)
            print(f"[figures] {ds}: done")
        except ValueError as e:
            print(f"[figures] {ds}: skipped ({e})")

    eda_path = Path(args.eda_stats)
    if eda_path.exists():
        with open(eda_path, encoding="utf-8") as f:
            stats = json.load(f)
        delta_map = {s["dataset"]: s["delta_mean"] for s in stats}
        try:
            res = geometry_delta_plot(
                args.runs, delta_map,
                str(Path(args.out) / "h1_geometry_delta.pdf"))
            print(f"[figures] H1 geometry x delta: Spearman rho="
                  f"{res['spearman_rho']:.2f} (p={res['spearman_p']:.3g}), "
                  f"Stouffer pooled p={res['stouffer_p']:.3g}")
        except ValueError as e:
            print(f"[figures] geometry x delta skipped: {e}")
    else:
        print("[figures] no EDA stats json — run scripts/run_eda.py for the "
              "geometry x delta figure")


if __name__ == "__main__":
    main()
