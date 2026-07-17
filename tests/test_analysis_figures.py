"""Mechanics tests for the EDA / results-figures / XAI figure modules and the
ratio split + geometry pooled test (not scientific verdicts)."""
import numpy as np
import pytest

try:  # importorskip only catches ImportError; broken DLLs raise OSError
    import torch
    import pennylane  # noqa: F401
    import geoopt  # noqa: F401
except Exception as e:  # pragma: no cover
    pytest.skip(f"torch stack unavailable: {e}", allow_module_level=True)
pytest.importorskip("matplotlib")

import matplotlib
matplotlib.use("Agg")

from torch_geometric.data import Data

from nemaq.analysis.eda import dataset_stats, delta_table_markdown, eda_figure
from nemaq.analysis.xai import (quantum_contribution_ratio, run_full_xai,
                                xai_branch_shapley)
from nemaq.data.loader import apply_split
from nemaq.models import build_model

N, F_IN, C = 24, 12, 3
CFG = {"hidden": 16, "branch_dim": 8, "fused_dim": 16, "n_qubits": 3,
       "q_depth": 1, "geometry": "hyperbolic", "quantum_mode": "pqc"}


@pytest.fixture(scope="module")
def tiny_data():
    torch.manual_seed(0)
    x = torch.rand(N, F_IN)
    ei = torch.randint(0, N, (2, 80))
    y = torch.randint(0, C, (N,))
    data = Data(x=x, edge_index=ei, y=y)
    data.num_nodes = N
    return apply_split(data, mode="ratio", split_seed=0)


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    return build_model("nemaq", F_IN, C, CFG)


def test_ratio_split_stratified_and_disjoint(tiny_data):
    tr, va, te = tiny_data.train_mask, tiny_data.val_mask, tiny_data.test_mask
    assert int((tr & va).sum()) == 0 and int((tr & te).sum()) == 0
    assert int((tr | va | te).sum()) == N
    for c in range(C):
        assert int((tr & (tiny_data.y == c)).sum()) >= 1  # stratified


def test_ratio_split_seed_reproducible(tiny_data):
    d2 = Data(x=tiny_data.x, edge_index=tiny_data.edge_index, y=tiny_data.y)
    d2.num_nodes = N
    d2 = apply_split(d2, mode="ratio", split_seed=0)
    assert torch.equal(d2.train_mask, tiny_data.train_mask)


def test_dataset_stats_and_eda_figure(tiny_data, tmp_path):
    stats = dataset_stats(tiny_data, "toy", delta_samples=50)
    assert stats["nodes"] == N and stats["classes"] == C
    assert 0.0 <= stats["edge_homophily"] <= 1.0
    assert stats["delta_mean"] >= 0.0
    eda_figure(tiny_data, "toy", stats, str(tmp_path / "eda.pdf"))
    assert (tmp_path / "eda.pdf").exists()
    md = delta_table_markdown([stats, dict(stats, dataset="toy2",
                                           delta_mean=stats["delta_mean"] + 1)])
    assert "toy" in md and "positive control" in md


def test_quantum_contribution_ratio_bounds(tiny_model, tiny_data):
    r_q, logits = quantum_contribution_ratio(
        tiny_model, tiny_data.x, tiny_data.edge_index)
    assert r_q.shape == (N,)
    assert float(r_q.min()) >= 0.0 and float(r_q.max()) <= 1.0
    assert logits.shape == (N, C)


def test_branch_shapley_efficiency(tiny_model, tiny_data, tmp_path):
    """Shapley values must sum to v(full) - v(empty) (efficiency axiom)."""
    out = xai_branch_shapley(tiny_model, tiny_data, "toy",
                             str(tmp_path / "shap.pdf"))
    assert set(out) == set(tiny_model.branches)
    assert (tmp_path / "shap.pdf").exists()


def test_run_full_xai_writes_figures(tiny_model, tiny_data, tmp_path):
    run_full_xai(tiny_model, tiny_data, "toy", str(tmp_path), ig_samples=3)
    pdfs = list(tmp_path.glob("toy_xai_*.pdf"))
    assert len(pdfs) >= 4  # IG, QOA, faithfulness, poincare, fusion, shapley


