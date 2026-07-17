"""Statistical protocol tests — the analysis code must be trustworthy
before it judges hypotheses."""
import pytest

pytest.importorskip("scipy")

from nemaq.analysis.stats import holm_bonferroni, paired_comparison


def test_paired_comparison_detects_clear_difference():
    a = {s: 0.85 + 0.01 * (s % 3) for s in range(10)}
    b = {s: 0.80 + 0.01 * (s % 3) for s in range(10)}
    r = paired_comparison(a, b)
    assert r["mean_diff"] == pytest.approx(0.05)
    assert r["wilcoxon_p"] < 0.05


def test_paired_comparison_requires_enough_seeds():
    with pytest.raises(ValueError):
        paired_comparison({0: 1.0, 1: 1.0}, {0: 0.9, 1: 0.9})


def test_holm_bonferroni_monotone_and_correct():
    res = holm_bonferroni({"h1": 0.001, "h2": 0.04, "h3": 0.30})
    assert res["h1"]["reject_null"] is True
    assert res["h3"]["reject_null"] is False
    ps = [res[k]["p_adjusted"] for k in ("h1", "h2", "h3")]
    assert ps == sorted(ps)
