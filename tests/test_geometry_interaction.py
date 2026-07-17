"""H1 pooled geometry test mechanics (torch-free: manifests + scipy only)."""
import numpy as np
import pytest

pytest.importorskip("scipy")

from nemaq.analysis.stats import geometry_interaction
from nemaq.utils.manifest import write_manifest


def _write_runs(root, ds, geo, base, n_seeds=8, sd=0.01, seed0=0):
    rng = np.random.default_rng(seed0)
    cfg = {"data": {"name": ds, "split": "public"},
           "model": {"name": "nemaq", "quantum_mode": "pqc",
                     "geometry": geo, "fusion_mode": "residual"}}
    for seed in range(n_seeds):
        write_manifest(root / f"{ds}_{geo}" / f"seed{seed}", cfg, seed,
                       {"test_acc": float(base + rng.normal(0, sd))})


def test_geometry_interaction_pooled(tmp_path):
    # dsa: low delta, real hyperbolic gain; dsb: high delta, no gain
    _write_runs(tmp_path, "dsa", "hyperbolic", 0.73)
    _write_runs(tmp_path, "dsa", "euclidean", 0.70, seed0=1)
    _write_runs(tmp_path, "dsb", "hyperbolic", 0.70, seed0=2)
    _write_runs(tmp_path, "dsb", "euclidean", 0.70, seed0=3)
    res = geometry_interaction(str(tmp_path), {"dsa": 0.1, "dsb": 1.5})
    assert set(res["per_dataset"]) == {"dsa", "dsb"}
    assert res["stouffer_p_one_sided"] < 0.05      # pooled effect detected
    assert res["spearman_delta_gain_rho"] < 0      # gain shrinks with delta
    assert res["per_dataset"]["dsa"]["wilcoxon_p_one_sided"] < 0.05


def test_geometry_interaction_requires_two_datasets(tmp_path):
    _write_runs(tmp_path, "dsa", "hyperbolic", 0.73)
    _write_runs(tmp_path, "dsa", "euclidean", 0.70)
    with pytest.raises(ValueError):
        geometry_interaction(str(tmp_path), {"dsa": 0.1})
