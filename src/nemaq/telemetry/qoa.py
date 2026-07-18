"""Quantum Observable Attribution (QOA) + faithfulness validation.

QOA (the ICEQT'26 paper's core method): per-node grad x input of the
predicted-class logit with respect to the measured quantum observables
(per-qubit Pauli-Z expectations), i.e. which observables drive the decision.

Referee B (ICEQT'26): "QOA is gradient saliency over intermediate
observables, but the paper does not provide faithfulness tests,
randomization checks, perturbation tests." This module supplies exactly
those three checks:

  1. faithfulness_masking  — zeroing the top-attributed observable must hurt
     the predicted logit more than zeroing the bottom-attributed one.
  2. perturbation_test     — noise injected on high-attribution observables
     must change predictions more than on low-attribution ones.
  3. randomization_check   — attributions must decay when the downstream
     model (fusion + head) is re-randomized (Adebayo et al. 2018 sanity check).
"""
import copy

import torch

from scipy.stats import spearmanr


def _observables(model, x):
    """Recompute the PQC observable vector [N, n_qubits] outside the graph."""
    qb = model.branches["q"]
    with torch.no_grad():
        obs = qb.qlayer(qb.angles(x).cpu()).to(x.device)
    return obs


def _logits_from_obs(model, x, edge_index, obs):
    out_q = model.branches["q"].project(obs)
    return model(x, edge_index, branch_override={"q": out_q})


def observable_attribution(model, x, edge_index):
    """Return (attr [N, n_qubits], obs [N, n_qubits]): grad x input of the
    predicted-class logit w.r.t. each measured observable."""
    model.eval()
    obs = _observables(model, x).requires_grad_(True)
    # enable_grad: callers (masking / perturbation checks) run under
    # @torch.no_grad(), which would otherwise leave sel without a graph.
    with torch.enable_grad():
        logits = _logits_from_obs(model, x, edge_index, obs)
        pred = logits.argmax(dim=-1)
        sel = logits[torch.arange(len(pred), device=logits.device), pred].sum()
        (grad,) = torch.autograd.grad(sel, obs)
    return (grad * obs).detach(), obs.detach()


@torch.no_grad()
def faithfulness_masking(model, x, edge_index, k: int = 1, mask=None) -> dict:
    """Zero top-k vs bottom-k attributed observables per node; report mean
    predicted-logit drop. Faithful attribution => drop_top >> drop_bottom."""
    attr, obs = observable_attribution(model, x, edge_index)
    base = _logits_from_obs(model, x, edge_index, obs)
    pred = base.argmax(dim=-1)
    idx = torch.arange(len(pred), device=base.device)
    order = attr.abs().argsort(dim=1)

    drops = {}
    for name, cols in {"top": order[:, -k:], "bottom": order[:, :k]}.items():
        obs_m = obs.clone().scatter_(1, cols, 0.0)
        masked = _logits_from_obs(model, x, edge_index, obs_m)
        d = base[idx, pred] - masked[idx, pred]
        drops[name] = float(d[mask].mean() if mask is not None else d.mean())
    drops["faithful"] = drops["top"] > drops["bottom"]
    return drops


@torch.no_grad()
def perturbation_test(model, x, edge_index, sigma: float = 0.25,
                      k: int = 1, mask=None, seed: int = 0) -> dict:
    """Gaussian noise on top-k vs bottom-k attributed observables; report
    prediction flip rates. Faithful => flip_top > flip_bottom."""
    attr, obs = observable_attribution(model, x, edge_index)
    base_pred = _logits_from_obs(model, x, edge_index, obs).argmax(dim=-1)
    order = attr.abs().argsort(dim=1)
    g = torch.Generator(device="cpu").manual_seed(seed)

    flips = {}
    for name, cols in {"top": order[:, -k:], "bottom": order[:, :k]}.items():
        noise = torch.randn(cols.shape, generator=g).to(obs.device) * sigma
        obs_p = obs.clone().scatter_add_(1, cols, noise)
        pred = _logits_from_obs(model, x, edge_index, obs_p).argmax(dim=-1)
        f = (pred != base_pred).float()
        flips[name] = float(f[mask].mean() if mask is not None else f.mean())
    flips["faithful"] = flips["top"] > flips["bottom"]
    return flips


def randomization_check(model, x, edge_index, seed: int = 0) -> dict:
    """Re-randomize fusion + head; Spearman correlation between original and
    randomized attributions should collapse toward 0 (attribution depends on
    the learned model, not on architecture artifacts)."""
    attr_orig, _ = observable_attribution(model, x, edge_index)

    # the last forward leaves non-leaf tensors (aux logits / branch outputs)
    # cached on the module; deepcopy chokes on them — clear first.
    if hasattr(model, "last_aux_logits"):
        model.last_aux_logits = {}
    if hasattr(model, "last_branch_outputs"):
        model.last_branch_outputs = {}
    rand = copy.deepcopy(model)
    torch.manual_seed(seed)
    for module in (rand.fusion, rand.head):
        for p in module.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.normal_(p, std=0.1)
    attr_rand, _ = observable_attribution(rand, x, edge_index)

    rho = spearmanr(attr_orig.flatten().cpu().numpy(),
                    attr_rand.flatten().cpu().numpy()).statistic
    return {"spearman_orig_vs_randomized": float(rho),
            "passes": abs(float(rho)) < 0.5}
