"""Pre-registered statistical protocol (addresses gap G2).

- Paired comparisons across seeds (same seeds/splits for both models).
- Wilcoxon signed-rank (non-parametric, n=10 seeds).
- Holm-Bonferroni correction over the hypothesis family.
- Cohen's d on paired differences as effect size.

Analysis consumes run manifests only (design rule #1).
"""
from collections import defaultdict

import numpy as np
from scipy import stats

from ..utils.manifest import read_manifests


def collect_scores(runs_root: str, metric: str = "test_acc") -> dict[tuple, dict[int, float]]:
    """Group manifest metrics by (dataset, model, split_mode), keyed by seed."""
    grouped: dict[tuple, dict[int, float]] = defaultdict(dict)
    for m in read_manifests(runs_root):
        cfg = m["config"]
        model_key = cfg["model"]["name"]
        if model_key == "nemaq":  # variant flags only matter for the hybrid
            model_key += (":" + cfg["model"].get("quantum_mode", "-")
                          + ":" + cfg["model"].get("geometry", "-")
                          + ":" + cfg["model"].get("fusion_mode", "residual"))
        key = (
            cfg["data"]["name"],
            model_key,
            cfg["data"].get("split", "public"),
        )
        grouped[key][m["seed"]] = m["metrics"][metric]
    return grouped


def paired_comparison(scores_a: dict[int, float], scores_b: dict[int, float]) -> dict:
    """Wilcoxon + Cohen's d on seed-paired scores. a = treatment, b = control."""
    seeds = sorted(set(scores_a) & set(scores_b))
    if len(seeds) < 5:
        raise ValueError(f"Only {len(seeds)} paired seeds; protocol requires >= 5 (target 10).")
    a = np.array([scores_a[s] for s in seeds])
    b = np.array([scores_b[s] for s in seeds])
    diff = a - b
    if np.allclose(diff, 0):
        w_p = 1.0
    else:
        w_p = float(stats.wilcoxon(a, b).pvalue)
    sd = diff.std(ddof=1)
    if sd > 0:
        d = float(diff.mean() / sd)
    else:
        # constant nonzero difference = deterministic effect, not zero effect
        d = float(np.sign(diff.mean()) * np.inf) if diff.mean() != 0 else 0.0
    return {
        "n_seeds": len(seeds),
        "mean_a": float(a.mean()), "std_a": float(a.std(ddof=1)),
        "mean_b": float(b.mean()), "std_b": float(b.std(ddof=1)),
        "mean_diff": float(diff.mean()),
        "wilcoxon_p": w_p,
        "cohens_d": d,
    }


def geometry_interaction(runs_root: str, delta_map: dict[str, float],
                         hyp_key: str = "nemaq:pqc:hyperbolic:residual",
                         euc_key: str = "nemaq:pqc:euclidean:residual") -> dict:
    """H1's pooled-power test for the geometry effect.

    The single-dataset hyperbolic-vs-Euclidean comparison is underpowered at
    n=10 seeds (Cora Phase 4: paired d=0.44 -> ~40 seeds needed alone). The
    registered remedy pools evidence instead:
      1. per-dataset paired Wilcoxon (one-sided, H1 direction: hyp > euc),
      2. Stouffer's Z across datasets (pooled directional evidence),
      3. Spearman(delta_mean, gain) across datasets (the trend H1 predicts:
         gain shrinks as delta grows).
    delta_map: {dataset: delta_mean}, from the EDA stats json.
    """
    grouped = collect_scores(runs_root)
    per_dataset, zs = {}, []
    for ds, delta in delta_map.items():
        sh = se = None
        for (d, mk, _sp), scores in grouped.items():
            if d != ds:
                continue
            if mk == hyp_key:
                sh = scores
            elif mk == euc_key:
                se = scores
        if sh is None or se is None:
            continue
        seeds = sorted(set(sh) & set(se))
        if len(seeds) < 5:
            continue
        a = np.array([sh[s] for s in seeds])
        b = np.array([se[s] for s in seeds])
        if np.allclose(a - b, 0):
            p_one = 0.5
        else:
            p_one = float(stats.wilcoxon(a, b, alternative="greater").pvalue)
        cmp = paired_comparison(sh, se)
        cmp.update({"delta_mean": delta, "wilcoxon_p_one_sided": p_one})
        per_dataset[ds] = cmp
        zs.append(stats.norm.isf(p_one))
    if len(per_dataset) < 2:
        raise ValueError("geometry_interaction needs the geometry pair on "
                         ">= 2 datasets")
    z_pooled = float(np.sum(zs) / np.sqrt(len(zs)))
    deltas = [per_dataset[d]["delta_mean"] for d in per_dataset]
    gains = [per_dataset[d]["mean_diff"] for d in per_dataset]
    rho = stats.spearmanr(deltas, gains)
    return {
        "per_dataset": per_dataset,
        "stouffer_z": z_pooled,
        "stouffer_p_one_sided": float(stats.norm.sf(z_pooled)),
        "spearman_delta_gain_rho": float(rho.statistic),
        "spearman_delta_gain_p": float(rho.pvalue),
    }


def holm_bonferroni(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Return per-hypothesis adjusted p and reject decision."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out, still_rejecting = {}, True
    running_max = 0.0
    for i, (name, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running_max = max(running_max, adj)  # enforce monotonicity
        reject = still_rejecting and running_max <= alpha
        if not reject:
            still_rejecting = False
        out[name] = {"p": p, "p_adjusted": running_max, "reject_null": reject}
    return out
