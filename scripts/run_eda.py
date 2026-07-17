"""EDA for all matrix datasets: per-dataset figure + delta-stratification
table/figure (fills PREREGISTRATION.md section 2).

Usage:
    python scripts/run_eda.py --datasets cora citeseer pubmed disease
"""
import argparse

from nemaq.analysis.eda import run_eda


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["cora", "citeseer", "pubmed", "disease"])
    ap.add_argument("--out", default="experiments/figures/eda")
    ap.add_argument("--root", default="data")
    args = ap.parse_args()
    stats = run_eda(args.datasets, args.out, args.root)
    print("\nDelta-stratification table (paste into PREREGISTRATION.md §2):\n")
    from nemaq.analysis.eda import delta_table_markdown
    print(delta_table_markdown(stats))


if __name__ == "__main__":
    main()
