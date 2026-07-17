"""Single training run.

Usage:
    python scripts/train.py --config configs/cora_nemaq.yaml --seed 0
"""
import argparse
from pathlib import Path

from nemaq.data.loader import apply_split, load_dataset
from nemaq.training.trainer import train_run
from nemaq.utils.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/runs")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if cfg.get("device") == "auto":
        cfg["device"] = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    ds = load_dataset(cfg["data"]["name"], cfg["data"].get("root", "data"))
    data = apply_split(
        ds[0],
        mode=cfg["data"].get("split", "public"),
        labels_per_class=cfg["data"].get("labels_per_class", 5),
        split_seed=cfg["data"].get("split_seed", 0),
    )

    run_dir = (Path(args.out) / Path(args.config).stem / f"seed{args.seed}")
    metrics = train_run(cfg, data, args.seed, run_dir)
    print(f"[{Path(args.config).stem} seed={args.seed}] "
          f"val={metrics['val_acc']:.4f} test={metrics['test_acc']:.4f} "
          f"epochs={metrics['epochs_run']}")


if __name__ == "__main__":
    main()
