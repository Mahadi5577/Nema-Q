"""XAI figure suite for NEMA-Q (port of the ICEQT'26 v9 XAI module to the
branch architecture, generalized to any dataset).

  xai_integrated_gradients — IG on input features per predicted class.
  xai_qoa_figures          — QOA heatmap (class x observable) + global ranking
                             (uses telemetry.qoa.observable_attribution).
  xai_qoa_faithfulness_fig — masking / perturbation / randomization panel
                             (the referee-B validation harness, visualized).
  xai_poincare             — Poincare-disk view of the geo branch (2D PCA of
                             the tangent output, exp-mapped to the disk).
  xai_fusion_decomposition — gradient-attributed per-node r_Q (quantum
                             contribution ratio) distributions.
  xai_branch_shapley       — exact Shapley values over branches (2^B
                             coalitions via disable_branch) per node/class.

All figures are dataset-generic: class count from labels, display names from
plotstyle.class_names_for.
"""
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from ..telemetry.qoa import (faithfulness_masking, observable_attribution,
                             perturbation_test, randomization_check)
from .plotstyle import (BRANCH_COLORS, BRANCH_LABELS, CLASS_COLORS,
                        apply_style, class_names_for)


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def _device(model) -> torch.device:
    return next(model.parameters()).device


def _class_grid(k: int) -> tuple[int, int]:
    cols = min(4, k)
    return math.ceil(k / cols), cols


def _branch_grads(model, x, edge_index):
    """Forward with every branch output substituted by a requires_grad leaf;
    backward the predicted-class logits. Returns per-branch |grad| L1 per
    node — the gradient-attributed branch sensitivity used for r_Q."""
    model.eval()
    with torch.no_grad():
        outs = {n: b(x, edge_index) for n, b in model.branches.items()}
    leaves = {n: o.detach().requires_grad_(True) for n, o in outs.items()}
    logits = model(x, edge_index, branch_override=leaves)
    pred = logits.argmax(dim=-1)
    sel = logits[torch.arange(len(pred), device=logits.device), pred].sum()
    grads = torch.autograd.grad(sel, list(leaves.values()))
    a = {n: g.abs().sum(dim=1) for n, g in zip(leaves, grads)}
    return a, logits.detach(), pred.detach()


def quantum_contribution_ratio(model, x, edge_index):
    """r_Q(i) = |grad wrt q| / sum over branches — gradient-attributed
    (decision sensitivity, not activation energy). Returns (r_q, logits)."""
    a, logits, _ = _branch_grads(model, x, edge_index)
    total = sum(a.values()) + 1e-8
    return (a["q"] / total if "q" in a else torch.zeros_like(total)), logits


# ── XAI-01: Integrated Gradients on input features ─────────────────────────

def xai_integrated_gradients(model, data, dataset: str, save_path: str,
                             n_steps: int = 32, n_samples: int = 70,
                             top_k: int = 20, seed: int = 42) -> np.ndarray:
    model.eval()
    dev = _device(model)
    x, ei, y = data.x, data.edge_index, data.y
    test_idx = data.test_mask.nonzero(as_tuple=True)[0]
    k = int(y.max()) + 1
    names = class_names_for(dataset, k)

    rng = np.random.default_rng(seed)
    sample = []
    per_class = max(1, n_samples // k)
    for c in range(k):
        idxs = _np(test_idx[(y[test_idx] == c).cpu()])
        if len(idxs):
            sample.extend(rng.choice(idxs, size=min(per_class, len(idxs)),
                                     replace=False).tolist())

    attr, preds = [], []
    alphas = torch.linspace(0, 1, n_steps, device=dev)
    for node in sample:
        with torch.no_grad():
            pred = int(model(x, ei).argmax(dim=-1)[node])
        preds.append(pred)
        grad_sum = torch.zeros(x.size(1), device=dev)
        for alpha in alphas:
            x_mod = (alpha * x).detach()
            row = x_mod[node].clone().requires_grad_(True)
            x_in = x_mod.index_copy(
                0, torch.tensor([node], device=dev), row.unsqueeze(0))
            logit = model(x_in, ei)[node, pred]
            (g,) = torch.autograd.grad(logit, row)
            grad_sum += g
        attr.append(_np(x[node] * grad_sum / n_steps))
    attr = np.array(attr)
    preds = np.array(preds)

    apply_style()
    rows, cols = _class_grid(k)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows),
                             squeeze=False)
    for c in range(k):
        ax = axes[c // cols][c % cols]
        mask = preds == c
        if mask.sum() == 0:
            ax.set_title(f"{names[c]} (no samples)")
            ax.axis("off")
            continue
        mean_abs = np.abs(attr[mask]).mean(axis=0)
        top = np.argsort(mean_abs)[-top_k:][::-1]
        ax.barh(range(len(top)), mean_abs[top],
                color=CLASS_COLORS[c % len(CLASS_COLORS)], alpha=0.85)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels([f"f{i}" for i in top], fontsize=7)
        ax.invert_yaxis()
        ax.set_title(f"{names[c]} (n={int(mask.sum())})", fontsize=10)
        ax.set_xlabel("Mean |IG|", fontsize=9)
        ax.grid(True, axis="x", linestyle=":", alpha=0.6)
    for j in range(k, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(f"{dataset} — XAI-01 Integrated Gradients: "
                 f"top-{top_k} input features per predicted class",
                 fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    return attr


# ── XAI-02: Quantum Observable Attribution figures ──────────────────────────

def xai_qoa_figures(model, data, dataset: str, save_path: str) -> np.ndarray:
    x, ei, y = data.x, data.edge_index, data.y
    test = data.test_mask
    k = int(y.max()) + 1
    names = class_names_for(dataset, k)
    nq = model.branches["q"].n_qubits
    obs_names = [f"$\\langle Z_{i} \\rangle$" for i in range(nq)]

    attr, _ = observable_attribution(model, x, ei)
    with torch.no_grad():
        preds = model(x, ei).argmax(dim=-1)
    attr_t = _np(attr[test].abs())
    preds_t = _np(preds[test])

    heat = np.zeros((k, nq))
    for c in range(k):
        m = preds_t == c
        if m.sum():
            heat[c] = attr_t[m].mean(axis=0)

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5),
                             gridspec_kw={"width_ratios": [1.8, 1]})
    ax = axes[0]
    im = ax.imshow(heat, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="Mean |grad × obs|")
    ax.set_xticks(range(nq))
    ax.set_xticklabels(obs_names, fontsize=10)
    ax.set_yticks(range(k))
    ax.set_yticklabels(names, fontsize=9)
    for c in range(k):
        for q in range(nq):
            ax.text(q, c, f"{heat[c, q]:.2f}", ha="center", va="center",
                    fontsize=8,
                    color="white" if heat[c, q] > heat.max() * 0.6 else "black")
    ax.set_title("QOA: attribution per class × observable")

    ax2 = axes[1]
    glob = attr_t.mean(axis=0)
    rank = np.argsort(glob)[::-1]
    ax2.barh(range(nq), glob[rank], color="#9467bd", alpha=0.85)
    ax2.set_yticks(range(nq))
    ax2.set_yticklabels([obs_names[r] for r in rank], fontsize=10)
    ax2.invert_yaxis()
    ax2.set_xlabel("Global mean |QOA|")
    ax2.set_title("Observable importance")
    ax2.grid(True, axis="x", linestyle=":", alpha=0.6)

    fig.suptitle(f"{dataset} — XAI-02 Quantum Observable Attribution",
                 fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    return attr_t


def xai_qoa_faithfulness_fig(model, data, dataset: str, save_path: str,
                             k_obs: int = 1) -> dict:
    """Referee-B validation harness as one panel: masking drops,
    perturbation flip rates, randomization correlation."""
    x, ei = data.x, data.edge_index
    test = data.test_mask
    masking = faithfulness_masking(model, x, ei, k=k_obs, mask=test)
    perturb = perturbation_test(model, x, ei, k=k_obs, mask=test)
    randch = randomization_check(model, x, ei)

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    ax = axes[0]
    ax.bar([0, 1], [masking["top"], masking["bottom"]],
           color=["#d62728", "#1f77b4"], alpha=0.85)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["mask top-attr", "mask bottom-attr"])
    ax.set_ylabel("Mean predicted-logit drop")
    ax.set_title(f"(a) Masking — faithful: {masking['faithful']}")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    ax = axes[1]
    ax.bar([0, 1], [perturb["top"], perturb["bottom"]],
           color=["#d62728", "#1f77b4"], alpha=0.85)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["perturb top-attr", "perturb bottom-attr"])
    ax.set_ylabel("Prediction flip rate")
    ax.set_title(f"(b) Perturbation — faithful: {perturb['faithful']}")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    ax = axes[2]
    rho = randch["spearman_orig_vs_randomized"]
    ax.bar([0], [abs(rho)], color="#2ca02c" if randch["passes"] else "#d62728",
           alpha=0.85)
    ax.axhline(0.5, color="black", linestyle="--", lw=1.2,
               label="pass threshold (|ρ| < 0.5)")
    ax.set_xticks([0])
    ax.set_xticklabels(["|Spearman ρ|\norig vs randomized"])
    ax.set_ylim(0, 1)
    ax.set_title(f"(c) Randomization — passes: {randch['passes']}")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    fig.suptitle(f"{dataset} — QOA faithfulness validation "
                 "(masking / perturbation / randomization)",
                 fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    return {"masking": masking, "perturbation": perturb,
            "randomization": randch}


# ── XAI-03: Poincare disk topology ──────────────────────────────────────────

def xai_poincare(model, data, dataset: str, save_path: str) -> None:
    if "geo" not in model.branches or not hasattr(model.branches["geo"],
                                                  "manifold"):
        print("[XAI-03] no hyperbolic geo branch — skipped")
        return
    x, ei, y = data.x, data.edge_index, data.y
    test_idx = data.test_mask.nonzero(as_tuple=True)[0]
    k = int(y.max()) + 1
    names = class_names_for(dataset, k)
    geo = model.branches["geo"]

    model.eval()
    with torch.no_grad():
        tangent = geo(x, ei)                      # [N, emb] tangent at origin
        logits = model(x, ei)
    r_q, _ = quantum_contribution_ratio(model, x, ei)

    # 2D PCA of tangent vectors, exp-mapped back onto the (2D) disk at the
    # learned curvature — a faithful "view" of the hyperbolic embedding.
    t_np = _np(tangent)
    t_c = t_np - t_np.mean(axis=0)
    _, _, vt = np.linalg.svd(t_c, full_matrices=False)
    t2 = torch.tensor(t_c @ vt[:2].T, dtype=torch.float32)
    manifold = geo.manifold
    with torch.no_grad():
        disk = _np(manifold.projx(manifold.expmap0(t2.to(tangent.device))))

    ti = _np(test_idx)
    pts = disk[ti]
    true_np = _np(y[test_idx])
    probs = _np(F.softmax(logits[test_idx], dim=1))
    preds_np = probs.argmax(axis=1)
    conf_np = probs.max(axis=1)
    rq_np = _np(r_q[test_idx])

    # class prototypes: tangent-space mean, exp-mapped (Frechet 1st-order)
    protos = {}
    y_np = _np(y)
    for c in range(k):
        m = y_np == c
        if m.sum() < 2:
            continue
        mu = torch.tensor((t_c[m]).mean(axis=0) @ vt[:2].T,
                          dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            protos[c] = _np(manifold.projx(
                manifold.expmap0(mu.to(tangent.device))))[0]

    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
    theta = np.linspace(0, 2 * np.pi, 300)
    panels = [
        ("True class", true_np, "class", True),
        ("Predicted class", preds_np, "class", True),
        ("Prediction confidence", conf_np, "RdYlGn", False),
        ("$r_Q$ (quantum contribution)", rq_np, "PuOr", False),
    ]
    for ax, (title, vals, cmap, show_proto) in zip(axes.flatten(), panels):
        ax.plot(np.cos(theta), np.sin(theta), "k-", lw=0.8, alpha=0.3)
        if cmap == "class":
            for c in range(k):
                m = vals == c
                if m.sum() == 0:
                    continue
                ax.scatter(pts[m, 0], pts[m, 1],
                           c=CLASS_COLORS[c % len(CLASS_COLORS)], s=6,
                           alpha=0.55, linewidths=0, label=names[c])
            if show_proto:
                for c, p in protos.items():
                    ax.scatter(p[0], p[1], c=CLASS_COLORS[c % len(CLASS_COLORS)],
                               s=160, marker="*", edgecolors="black",
                               linewidths=0.8, zorder=10)
            ax.legend(fontsize=6.5, markerscale=2, loc="upper right",
                      framealpha=0.85)
        else:
            sc = ax.scatter(pts[:, 0], pts[:, 1], c=vals, cmap=cmap, s=6,
                            alpha=0.65, linewidths=0)
            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=11, weight="bold")
    fig.suptitle(
        f"{dataset} — XAI-03 Poincaré disk view of the geo branch\n"
        "(2D PCA of tangent output, exp-mapped at learned curvature "
        f"c={geo.curvature:.3f}; stars = class prototypes)",
        fontsize=12, weight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


# ── XAI-04: per-node fusion decomposition ───────────────────────────────────

def xai_fusion_decomposition(model, data, dataset: str,
                             save_path: str) -> np.ndarray:
    x, ei, y = data.x, data.edge_index, data.y
    test_idx = data.test_mask.nonzero(as_tuple=True)[0]
    k = int(y.max()) + 1
    names = class_names_for(dataset, k)

    r_q, logits = quantum_contribution_ratio(model, x, ei)
    rq = _np(r_q[test_idx])
    probs = _np(F.softmax(logits[test_idx], dim=1))
    conf = probs.max(axis=1)
    true_np = _np(y[test_idx])

    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0][0]
    groups = [rq[true_np == c] for c in range(k)]
    present = [c for c in range(k) if len(groups[c])]
    parts = ax.violinplot([groups[c] for c in present], positions=present,
                          showmedians=True)
    for pc, c in zip(parts["bodies"], present):
        pc.set_facecolor(CLASS_COLORS[c % len(CLASS_COLORS)])
        pc.set_alpha(0.65)
    ax.axhline(rq.mean(), color="navy", linestyle=":", lw=1.5,
               label=f"mean $r_Q$ = {rq.mean():.3f}")
    ax.set_xticks(range(k))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("$r_Q$")
    ax.set_ylim(0, 1)
    ax.set_title("(a) $r_Q$ by true class (gradient-attributed)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    ax = axes[0][1]
    ax.hist(rq, bins=40, color="#9467bd", alpha=0.8, edgecolor="black",
            linewidth=0.4)
    ax.axvline(rq.mean(), color="navy", linestyle=":", lw=2,
               label=f"mean = {rq.mean():.3f}")
    ax.set_xlabel("$r_Q$")
    ax.set_ylabel("Count")
    ax.set_title("(b) $r_Q$ distribution (test nodes)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    ax = axes[1][0]
    for c in range(k):
        m = probs.argmax(axis=1) == c
        if m.sum() == 0:
            continue
        ax.scatter(conf[m], rq[m], c=CLASS_COLORS[c % len(CLASS_COLORS)],
                   s=8, alpha=0.45, linewidths=0, label=names[c])
    ax.set_xlabel("Prediction confidence")
    ax.set_ylabel("$r_Q$")
    ax.set_title("(c) $r_Q$ vs confidence")
    ax.legend(fontsize=6.5, ncol=2)
    ax.grid(True, linestyle=":", alpha=0.5)

    ax = axes[1][1]
    means = [groups[c].mean() if len(groups[c]) else 0 for c in range(k)]
    stds = [groups[c].std() if len(groups[c]) else 0 for c in range(k)]
    ax.bar(range(k), means, yerr=stds, color=CLASS_COLORS[:k], alpha=0.85,
           capsize=4)
    ax.set_xticks(range(k))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Mean $r_Q$ ± std")
    ax.set_ylim(0, 1)
    ax.set_title("(d) Per-class mean $r_Q$")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    fig.suptitle(f"{dataset} — XAI-04 Fusion decomposition: "
                 "quantum contribution per node", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    return rq


# ── XAI-05: exact branch Shapley ────────────────────────────────────────────

def xai_branch_shapley(model, data, dataset: str, save_path: str) -> dict:
    """Exact per-node Shapley value of each branch for the predicted-class
    logit. Coalitions = subsets of branches present (absent branch output is
    zeroed via disable_branch semantics), 2^B forwards — exact, no sampling.
    Replaces the v9 KernelSHAP-on-fused-vector (fused dims are no longer
    individually meaningful after per-branch projection)."""
    from itertools import combinations

    x, ei, y = data.x, data.edge_index, data.y
    test = data.test_mask
    k = int(y.max()) + 1
    names = class_names_for(dataset, k)
    branches = list(model.branches)
    B = len(branches)

    model.eval()
    with torch.no_grad():
        outs = {n: b(x, ei) for n, b in model.branches.items()}
        zeros = {n: torch.zeros_like(o) for n, o in outs.items()}

        def value(subset: frozenset) -> torch.Tensor:
            ov = {n: (outs[n] if n in subset else zeros[n]) for n in branches}
            lg = model(x, ei, branch_override=ov)
            return lg

        full = value(frozenset(branches))
        pred = full.argmax(dim=-1)
        idx = torch.arange(len(pred), device=pred.device)

        vals = {}
        for r in range(B + 1):
            for sub in combinations(branches, r):
                vals[frozenset(sub)] = value(frozenset(sub))[idx, pred]

        phi = {}
        for b in branches:
            acc = torch.zeros_like(vals[frozenset()])
            others = [n for n in branches if n != b]
            for r in range(len(others) + 1):
                w = (math.factorial(r) * math.factorial(B - r - 1)
                     / math.factorial(B))
                for sub in combinations(others, r):
                    s = frozenset(sub)
                    acc = acc + w * (vals[s | {b}] - vals[s])
            phi[b] = acc

    true_np = _np(y[test])
    phi_np = {b: _np(p[test]) for b, p in phi.items()}

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5),
                             gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    heat = np.zeros((k, B))
    for c in range(k):
        m = true_np == c
        for j, b in enumerate(branches):
            if m.sum():
                heat[c, j] = np.abs(phi_np[b][m]).mean()
    im = ax.imshow(heat, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax, label="Mean |Shapley value|")
    ax.set_xticks(range(B))
    ax.set_xticklabels([BRANCH_LABELS.get(b, b) for b in branches], fontsize=9)
    ax.set_yticks(range(k))
    ax.set_yticklabels(names, fontsize=9)
    for c in range(k):
        for j in range(B):
            ax.text(j, c, f"{heat[c, j]:.2f}", ha="center", va="center",
                    fontsize=8,
                    color="white" if heat[c, j] > heat.max() * 0.6 else "black")
    ax.set_title("Exact branch Shapley: class × branch")

    ax = axes[1]
    parts = ax.violinplot([phi_np[b] for b in branches],
                          positions=range(B), showmedians=True)
    for pc, b in zip(parts["bodies"], branches):
        pc.set_facecolor(BRANCH_COLORS.get(b, "#7f7f7f"))
        pc.set_alpha(0.7)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(range(B))
    ax.set_xticklabels([BRANCH_LABELS.get(b, b) for b in branches],
                       fontsize=8, rotation=15)
    ax.set_ylabel("Shapley value (predicted logit)")
    ax.set_title("Per-node Shapley distribution")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)

    fig.suptitle(f"{dataset} — XAI-05 Exact branch Shapley decomposition",
                 fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    return {b: float(np.abs(v).mean()) for b, v in phi_np.items()}


# ── master runner ───────────────────────────────────────────────────────────

def run_full_xai(model, data, dataset: str,
                 out_dir: str = "experiments/figures/xai",
                 ig_samples: int = 70) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pre = str(out / f"{dataset}_xai")

    print(f"[XAI] {dataset}: integrated gradients …")
    xai_integrated_gradients(model, data, dataset, pre + "_01_intgrad.pdf",
                             n_samples=ig_samples)
    if "q" in model.branches and hasattr(model.branches["q"], "qlayer"):
        print(f"[XAI] {dataset}: QOA figures …")
        xai_qoa_figures(model, data, dataset, pre + "_02_qoa.pdf")
        print(f"[XAI] {dataset}: QOA faithfulness …")
        res = xai_qoa_faithfulness_fig(model, data, dataset,
                                       pre + "_02_qoa_faithfulness.pdf")
        print(f"       masking faithful={res['masking']['faithful']} "
              f"perturbation faithful={res['perturbation']['faithful']} "
              f"randomization passes={res['randomization']['passes']}")
    print(f"[XAI] {dataset}: Poincaré disk …")
    xai_poincare(model, data, dataset, pre + "_03_poincare.pdf")
    if "q" in model.branches:
        print(f"[XAI] {dataset}: fusion decomposition …")
        xai_fusion_decomposition(model, data, dataset,
                                 pre + "_04_fusion.pdf")
    print(f"[XAI] {dataset}: branch Shapley …")
    xai_branch_shapley(model, data, dataset, pre + "_05_shapley.pdf")
    print(f"[XAI] {dataset}: figures in {out}")
