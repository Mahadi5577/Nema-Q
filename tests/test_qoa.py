"""QOA faithfulness harness tests (mechanics, not scientific verdicts)."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pennylane")
pytest.importorskip("geoopt")

from nemaq.models import build_model
from nemaq.telemetry.qoa import (faithfulness_masking, observable_attribution,
                                 perturbation_test, randomization_check)

N, F_IN, C = 20, 12, 3
CFG = {"hidden": 16, "branch_dim": 8, "fused_dim": 16, "n_qubits": 3,
       "q_depth": 1, "geometry": "euclidean", "quantum_mode": "pqc"}


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    x = torch.randn(N, F_IN)
    ei = torch.randint(0, N, (2, 60))
    model = build_model("nemaq", F_IN, C, CFG)
    return model, x, ei


def test_attribution_shape_and_grad(setup):
    model, x, ei = setup
    attr, obs = observable_attribution(model, x, ei)
    assert attr.shape == (N, CFG["n_qubits"])
    assert not attr.requires_grad  # detached result


def test_faithfulness_masking_returns_both_drops(setup):
    model, x, ei = setup
    d = faithfulness_masking(model, x, ei, k=1)
    assert set(d) == {"top", "bottom", "faithful"}


def test_perturbation_reports_flip_rates(setup):
    model, x, ei = setup
    f = perturbation_test(model, x, ei, sigma=0.5, k=1)
    assert 0.0 <= f["top"] <= 1.0 and 0.0 <= f["bottom"] <= 1.0


def test_randomization_check_runs(setup):
    model, x, ei = setup
    r = randomization_check(model, x, ei)
    assert -1.0 <= r["spearman_orig_vs_randomized"] <= 1.0


def test_trunk_only_ablation_builds_and_runs():
    cfg = dict(CFG, geometry="off", quantum_mode="off")
    model = build_model("nemaq", F_IN, C, cfg)
    assert list(model.branches) == ["bypass"]
    out = model(torch.randn(N, F_IN), torch.randint(0, N, (2, 40)))
    assert out.shape == (N, C)
