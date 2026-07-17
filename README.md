# NEMA-Q

Instrumentation-first study of a hybrid hyperbolic + variational-quantum graph
neural network. Research plan: [RESEARCH_PIPELINE.md](RESEARCH_PIPELINE.md).
Registered analysis: [experiments/PREREGISTRATION.md](experiments/PREREGISTRATION.md).

## Setup

```bash
python -m venv .venv && .venv/Scripts/activate   # Windows
pip install -e .[dev]
pytest
```

## Quick start

```bash
# baseline sanity (Gate 2 target: ~81.5% on Cora public split)
python scripts/train.py --config configs/cora_gcn.yaml --seed 0

# full hybrid
python scripts/train.py --config configs/cora_nemaq.yaml --seed 0

# dataset hyperbolicity table (H1 stratification)
python scripts/compute_hyperbolicity.py --datasets cora citeseer pubmed texas

# evaluation matrix (resumable; skips existing manifests)
python scripts/run_matrix.py --configs configs/cora_gcn.yaml configs/cora_nemaq.yaml \
    configs/ablations/nemaq_surrogate.yaml --seeds 10
```

## Design rules

1. Every run writes a `manifest.json`; analysis reads manifests only.
2. Every control (surrogate, frozen-random, Euclidean) is a config swap, not a code fork.
3. Parameter matching between PQC and surrogate is asserted in tests.
4. Telemetry (per-branch gradients, fusion gates) is always on.

## Layout

```
configs/          experiment configs (base + overrides)
src/nemaq/
  data/           loaders, splits, δ-hyperbolicity
  models/         baselines, hyperbolic encoder, PQC, surrogate, fusion, NEMA-Q
  telemetry/      gradient variance (H3), expressibility, attribution (H4)
  training/       trainer (early stopping, manifests)
  analysis/       Wilcoxon / Holm–Bonferroni / effect sizes (pre-registered)
scripts/          train.py, run_matrix.py, compute_hyperbolicity.py
tests/            component + protocol tests (Gates 1–3)
experiments/      PREREGISTRATION.md, runs/ (gitignored)
```
