"""Paper results figures. All consume run manifests only (design rule #1).

  forest_plot           — per-model mean test acc with 95% CI, one dataset.
  component_waterfall   — the component-accounting story: GCN -> scaffold ->
                          +frozen PQC -> trained PQC, as cumulative deficits.
  paired_diff_plot      — seed-paired slope chart for any two models.
  stability_plot        — across-seed std per model (fusion stability claim).
  geometry_delta_plot   — H1 headline: paired (hyperbolic - Euclidean) gain
                          vs dataset delta, with Spearman trend.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sps

from .plotstyle import MODEL_LABELS, apply_style
from .stats import collect_scores


def _label(key: str) -> str:
    return MODEL_LABELS.get(key, key)


def _series(grouped, dataset, model_key, split="public"):
    for (ds, mk, sp), scores in grouped.items():
        if ds == dataset and mk == model_key and sp == split:
            return scores
    return None


def _mean_ci(vals: np.ndarray, conf: float = 0.95) -> tuple[float, float]:
    m = vals.mean()
    if len(vals) < 2:
        return m, 0.0
    half = sps.t.ppf(0.5 + conf / 2, len(vals) - 1) * vals.std(ddof=1) / np.sqrt(len(vals))
    return m, half


def forest_plot(runs_root: str, dataset: str, save_path: str,
                split: str = "public") -> None:
    grouped = collect_scores(runs_root)
    rows = []
    for (ds, mk, sp), scores in sorted(grouped.items()):
        if ds != dataset or sp != split:
            continue
        vals = np.array(list(scores.values()))
        m, half = _mean_ci(vals)
        rows.append((_label(mk), m, half, len(vals)))
    if not rows:
        raise ValueError(f"No manifests for dataset={dataset} split={split}")
    rows.sort(key=lambda r: r[1])

    apply_style()
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(rows) + 2))
    ys = np.arange(len(rows))
    for y, (name, m, half, n) in zip(ys, rows):
        ax.errorbar(m, y, xerr=half, fmt="o", color="#1f77b4",
                    ecolor="#1f77b4", capsize=4, ms=6)
        ax.text(m + half + 0.004, y, f"{m:.3f} ± {half:.3f} (n={n})",
                va="center", fontsize=8)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=10)
    ax.set_xlabel("Test accuracy (mean, 95% CI)")
    ax.set_title(f"{dataset} — model comparison ({split} split)")
    ax.grid(True, axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


WATERFALL_STAGES = [
    ("gcn", "GCN\n(reference)"),
    ("nemaq:off:off:residual", "+ scaffold\n(trunk-only)"),
    ("nemaq:frozen_random:hyperbolic:residual", "+ frozen\nrandom PQC"),
    ("nemaq:pqc:hyperbolic:residual", "+ trained\nPQC"),
]


def component_waterfall(runs_root: str, dataset: str, save_path: str,
                        split: str = "public") -> None:
    """Cumulative accuracy at each architectural stage, with the per-stage
    paired deficit annotated (the paper's component-accounting figure)."""
    grouped = collect_scores(runs_root)
    series = []
    for key, label in WATERFALL_STAGES:
        s = _series(grouped, dataset, key, split)
        if s is None:
            raise ValueError(f"Missing runs for {key} on {dataset}")
        series.append((label, s))

    apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = np.arange(len(series))
    means, halves = [], []
    for _, s in series:
        m, h = _mean_ci(np.array(list(s.values())))
        means.append(m)
        halves.append(h)
    colors = ["#7f7f7f", "#ff7f0e", "#2ca02c", "#9467bd"]
    ax.bar(xs, means, yerr=halves, color=colors, alpha=0.85, capsize=5)

    # paired per-stage deltas (same seeds)
    for i in range(1, len(series)):
        prev, cur = series[i - 1][1], series[i][1]
        seeds = sorted(set(prev) & set(cur))
        diff = np.array([cur[s] - prev[s] for s in seeds])
        p = sps.wilcoxon(diff).pvalue if not np.allclose(diff, 0) else 1.0
        ax.annotate(
            f"{diff.mean() * 100:+.1f} pts\np={p:.3g}",
            xy=((xs[i - 1] + xs[i]) / 2, max(means[i - 1], means[i]) + 0.012),
            ha="center", fontsize=9, color="#d62728")
    ax.set_xticks(xs)
    ax.set_xticklabels([lbl for lbl, _ in series], fontsize=10)
    ax.set_ylabel("Test accuracy")
    lo = min(means) - 0.05
    ax.set_ylim(lo, max(means) + 0.05)
    ax.set_title(f"{dataset} — component accounting\n"
                 "(paired per-stage deficits, Wilcoxon p)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def paired_diff_plot(runs_root: str, dataset: str, model_a: str, model_b: str,
                     save_path: str, split: str = "public") -> None:
    """Slope chart: per-seed accuracy of two models + paired diff histogram."""
    grouped = collect_scores(runs_root)
    sa = _series(grouped, dataset, model_a, split)
    sb = _series(grouped, dataset, model_b, split)
    if sa is None or sb is None:
        raise ValueError(f"Missing runs for {model_a} or {model_b} on {dataset}")
    seeds = sorted(set(sa) & set(sb))
    a = np.array([sa[s] for s in seeds])
    b = np.array([sb[s] for s in seeds])
    diff = a - b
    p = sps.wilcoxon(a, b).pvalue if not np.allclose(diff, 0) else 1.0

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5),
                             gridspec_kw={"width_ratios": [1.2, 1]})
    ax = axes[0]
    for i, s in enumerate(seeds):
        ax.plot([0, 1], [b[i], a[i]], "-o", ms=4, alpha=0.6,
                color="#2ca02c" if diff[i] > 0 else "#d62728")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([_label(model_b), _label(model_a)], fontsize=10)
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"{dataset}: per-seed paired comparison")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    ax = axes[1]
    ax.hist(diff, bins=max(5, len(seeds) // 2), color="#1f77b4",
            alpha=0.8, edgecolor="black", linewidth=0.4)
    ax.axvline(0, color="black", lw=1)
    ax.axvline(diff.mean(), color="#d62728", linestyle="--", lw=1.5,
               label=f"mean {diff.mean() * 100:+.1f} pts")
    ax.set_xlabel(f"{_label(model_a)} − {_label(model_b)}")
    ax.set_title(f"Paired diff (n={len(seeds)}, Wilcoxon p={p:.3g})")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def stability_plot(runs_root: str, dataset: str, save_path: str,
                   split: str = "public") -> None:
    """Across-seed std per model — the fusion-stability claim in one figure."""
    grouped = collect_scores(runs_root)
    rows = []
    for (ds, mk, sp), scores in sorted(grouped.items()):
        if ds != dataset or sp != split:
            continue
        vals = np.array(list(scores.values()))
        rows.append((_label(mk), vals.std(ddof=1), len(vals)))
    rows.sort(key=lambda r: r[1])

    apply_style()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = np.arange(len(rows))
    colors = ["#d62728" if "softmax" in n.lower() or "surrogate" in n.lower()
              else "#1f77b4" for n, _, _ in rows]
    ax.bar(xs, [r[1] for r in rows], color=colors, alpha=0.85)
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in rows], rotation=30, ha="right",
                       fontsize=9)
    ax.set_ylabel("Across-seed std of test accuracy")
    ax.set_title(f"{dataset} — seed stability by model")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def geometry_delta_plot(runs_root: str, delta_map: dict[str, float],
                        save_path: str,
                        hyp_key: str = "nemaq:pqc:hyperbolic:residual",
                        euc_key: str = "nemaq:pqc:euclidean:residual") -> dict:
    """H1 figure: per-dataset paired (hyperbolic - Euclidean) gain vs delta.

    delta_map: {dataset: delta_mean} (from EDA stats json).
    Returns the underlying numbers, including Spearman(delta, gain) and a
    Stouffer-combined p across datasets — the pooled-power answer to the
    single-dataset underpowering of the geometry comparison.
    """
    grouped = collect_scores(runs_root)
    points = []
    for ds, delta in delta_map.items():
        # match whatever split the runs used for this dataset
        sh = se = None
        for (d, mk, sp), scores in grouped.items():
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
        diff = np.array([sh[s] - se[s] for s in seeds])
        t, p_two = sps.ttest_rel([sh[s] for s in seeds], [se[s] for s in seeds])
        m, half = _mean_ci(diff)
        points.append({"dataset": ds, "delta": delta, "gain": m, "ci": half,
                       "n": len(seeds), "t": float(t), "p": float(p_two)})
    if len(points) < 2:
        raise ValueError("Need geometry-pair runs on >= 2 datasets")

    deltas = np.array([p["delta"] for p in points])
    gains = np.array([p["gain"] for p in points])
    rho, rho_p = sps.spearmanr(deltas, gains)
    # Stouffer: one-sided z per dataset (H1 direction: gain > 0), pooled
    zs = np.array([sps.norm.isf(sps.t.sf(p["t"], p["n"] - 1)) for p in points])
    z_pooled = zs.sum() / np.sqrt(len(zs))
    p_pooled = float(sps.norm.sf(z_pooled))

    apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.axhline(0, color="black", lw=1)
    ax.errorbar(deltas, gains, yerr=[p["ci"] for p in points], fmt="o",
                ms=7, capsize=4, color="#1f77b4")
    for p in points:
        ax.annotate(f"{p['dataset']}\n(n={p['n']})", (p["delta"], p["gain"]),
                    xytext=(8, 6), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Sampled Gromov $\\delta$ (mean)")
    ax.set_ylabel("Paired gain: hyperbolic − Euclidean (test acc)")
    ax.set_title("H1: geometry gain vs hyperbolicity\n"
                 f"Spearman $\\rho$={rho:.2f} (p={rho_p:.3g});  "
                 f"pooled one-sided p={p_pooled:.3g} (Stouffer)")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    return {"points": points, "spearman_rho": float(rho),
            "spearman_p": float(rho_p), "stouffer_p": p_pooled}


def make_all(runs_root: str, dataset: str, out_dir: str,
             split: str = "public") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    forest_plot(runs_root, dataset, str(out / f"{dataset}_forest.pdf"), split)
    stability_plot(runs_root, dataset, str(out / f"{dataset}_stability.pdf"), split)
    try:
        component_waterfall(runs_root, dataset,
                            str(out / f"{dataset}_waterfall.pdf"), split)
    except ValueError as e:
        print(f"[figures] waterfall skipped for {dataset}: {e}")
    for a, b, tag in [
        ("nemaq:pqc:hyperbolic:residual", "nemaq:pqc:euclidean:residual",
         "geometry"),
        ("nemaq:frozen_random:hyperbolic:residual",
         "nemaq:pqc:hyperbolic:residual", "frozen_vs_trained"),
    ]:
        try:
            paired_diff_plot(runs_root, dataset, a, b,
                             str(out / f"{dataset}_paired_{tag}.pdf"), split)
        except ValueError as e:
            print(f"[figures] paired {tag} skipped for {dataset}: {e}")
