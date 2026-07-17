"""Train one NEMA-Q run and produce the full XAI figure suite for the paper.

Usage:
    python scripts/run_xai.py --config configs/cora_nemaq.yaml --seed 2
Seed 2 was Cora's median-accuracy seed in the Phase-4 matrix; pick per
dataset. The trained run's manifest is written like any matrix run
(resumable / provenance-consistent).
"""
import argparse
from pathlib import Path

import torch

from nemaq.analysis.xai import run_full_xai
from nemaq.data.loader import apply_split, load_dataset
from nemaq.training.trainer import train_run
from nemaq.utils.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--out", default="experiments/figures/xai")
    ap.add_argument("--runs", default="experiments/runs")
    ap.add_argument("--ig-samples", type=int, default=70)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if cfg.get("device") == "auto":
        cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    ds_name = cfg["data"]["name"]
    ds = load_dataset(ds_name, cfg["data"].get("root", "data"))
    data = apply_split(
        ds[0].clone(),
        mode=cfg["data"].get("split", "public"),
        labels_per_class=cfg["data"].get("labels_per_class", 5),
        split_seed=cfg["data"].get("split_seed", 0),
    )
    run_dir = Path(args.runs) / (Path(args.config).stem + "_xai") / f"seed{args.seed}"
    metrics, model, data = train_run(cfg, data, args.seed, run_dir,
                                     return_model=True)
    print(f"[XAI] trained {ds_name} seed={args.seed}: "
          f"test={metrics['test_acc']:.4f}")
    run_full_xai(model, data, ds_name, args.out, ig_samples=args.ig_samples)


if __name__ == "__main__":
    main()
