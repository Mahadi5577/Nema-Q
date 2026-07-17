"""Run the full evaluation matrix (Phase 5).

Iterates configs x seeds; skips runs whose manifest already exists
(resumable). All results land as manifests under experiments/runs/ —
analysis reads only those.

Usage:
    python scripts/run_matrix.py --configs configs/cora_gcn.yaml configs/cora_nemaq.yaml \
        --seeds 10
"""
import argparse
from pathlib import Path

from nemaq.data.loader import apply_split, load_dataset
from nemaq.training.trainer import train_run
from nemaq.utils.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--out", default="experiments/runs")
    args = ap.parse_args()

    for cfg_path in args.configs:
        cfg = load_config(cfg_path)
        if cfg.get("device") == "auto":
            cfg["device"] = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        ds = load_dataset(cfg["data"]["name"], cfg["data"].get("root", "data"))
        for seed in range(args.seeds):
            run_dir = Path(args.out) / Path(cfg_path).stem / f"seed{seed}"
            if (run_dir / "manifest.json").exists():
                print(f"skip {run_dir} (manifest exists)")
                continue
            data = apply_split(
                ds[0].clone(),
                mode=cfg["data"].get("split", "public"),
                labels_per_class=cfg["data"].get("labels_per_class", 5),
                split_seed=cfg["data"].get("split_seed", 0),
            )
            m = train_run(cfg, data, seed, run_dir)
            print(f"[{Path(cfg_path).stem} seed={seed}] test={m['test_acc']:.4f}")


if __name__ == "__main__":
    main()
