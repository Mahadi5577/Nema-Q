"""Component unit tests (Gate 3): shapes, gradient flow, control swaps."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from nemaq.models import build_model

N, F_IN, C = 20, 12, 3


@pytest.fixture
def tiny_graph():
    torch.manual_seed(0)
    x = torch.randn(N, F_IN)
    edge_index = torch.randint(0, N, (2, 60))
    y = torch.randint(0, C, (N,))
    return x, edge_index, y


@pytest.mark.parametrize("name", ["gcn", "gat", "mlp"])
def test_baselines_forward_backward(tiny_graph, name):
    x, ei, y = tiny_graph
    model = build_model(name, F_IN, C, {"hidden": 16})
    out = model(x, ei)
    assert out.shape == (N, C)
    torch.nn.functional.cross_entropy(out, y).backward()
    assert any(p.grad is not None for p in model.parameters())


def test_hyperbolic_encoder_maps_are_inverse():
    geoopt = pytest.importorskip("geoopt")
    from nemaq.models.hyperbolic import HyperbolicEncoder
    enc = HyperbolicEncoder(F_IN, 16, 8)
    v = torch.randn(5, F_IN) * 0.1
    on_ball = enc.manifold.projx(enc.manifold.expmap0(v))
    back = enc.manifold.logmap0(on_ball)
    assert torch.allclose(v, back, atol=1e-4)


@pytest.mark.parametrize("qmode", ["pqc", "surrogate", "frozen_random", "off"])
def test_nemaq_quantum_mode_swaps(tiny_graph, qmode):
    pytest.importorskip("pennylane")
    pytest.importorskip("geoopt")
    x, ei, y = tiny_graph
    cfg = {"hidden": 16, "branch_dim": 8, "fused_dim": 16,
           "n_qubits": 3, "q_depth": 1, "geometry": "euclidean",
           "quantum_mode": qmode}
    model = build_model("nemaq", F_IN, C, cfg)
    out = model(x, ei)
    assert out.shape == (N, C)
    torch.nn.functional.cross_entropy(out, y).backward()


def test_leave_branch_out_changes_output(tiny_graph):
    pytest.importorskip("pennylane")
    x, ei, _ = tiny_graph
    cfg = {"hidden": 16, "branch_dim": 8, "fused_dim": 16,
           "n_qubits": 3, "q_depth": 1, "geometry": "euclidean",
           "quantum_mode": "pqc"}
    model = build_model("nemaq", F_IN, C, cfg)
    model.eval()
    with torch.no_grad():
        full = model(x, ei)
        ablated = model(x, ei, disable_branch="q")
    assert not torch.allclose(full, ablated)
