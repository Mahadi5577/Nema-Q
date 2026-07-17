"""Residual-trunk fusion guarantees: training starts approximately at the
trunk (graceful-degradation floor), softmax ablation starts uniform."""
import pytest

torch = pytest.importorskip("torch")

from nemaq.models.fusion import GatedFusion


def test_residual_fusion_starts_near_trunk():
    torch.manual_seed(0)
    f = GatedFusion({"bypass": 8, "q": 8, "geo": 8}, 16, mode="residual")
    outs = {n: torch.randn(5, 8) for n in ("bypass", "q", "geo")}
    fused = f(outs)
    trunk = f.proj["bypass"](outs["bypass"])
    g0 = torch.sigmoid(torch.tensor(-2.0))
    expected = trunk + g0 * f.proj["q"](outs["q"]) + g0 * f.proj["geo"](outs["geo"])
    assert torch.allclose(fused, expected, atol=1e-5)
    # residual contribution starts small relative to trunk
    assert float(f.last_gates.mean()) == pytest.approx(float(g0), abs=1e-4)


def test_softmax_fusion_starts_uniform():
    torch.manual_seed(0)
    f = GatedFusion({"a": 8, "b": 8}, 16, mode="softmax")
    f({n: torch.randn(5, 8) for n in ("a", "b")})
    assert torch.allclose(f.last_gates, torch.full_like(f.last_gates, 0.5))


def test_residual_requires_trunk_branch():
    with pytest.raises(ValueError):
        GatedFusion({"a": 8, "b": 8}, 16, mode="residual", trunk="bypass")


def test_gate_dict_excludes_trunk_in_residual_mode():
    f = GatedFusion({"bypass": 8, "q": 8}, 16, mode="residual")
    f({n: torch.randn(3, 8) for n in ("bypass", "q")})
    assert set(f.gate_dict()) == {"q"}
