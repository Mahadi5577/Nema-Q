"""Per-node, per-branch attribution.

Two instruments:
  1. leave_branch_out — per-node change in predicted-class logit when one
     branch is zeroed. Ground truth for branch usefulness.
  2. gate_attribution — fusion gate weight per node per branch. Cheap proxy.

H4 = Spearman correlation between (1) and (2); computed in analysis.
"""
import torch

from scipy.stats import spearmanr


@torch.no_grad()
def leave_branch_out(model, x, edge_index, mask=None) -> dict[str, torch.Tensor]:
    """Return per-branch tensor of logit drops (positive = branch helped)."""
    model.eval()
    full = model(x, edge_index)
    pred = full.argmax(dim=-1)
    idx = torch.arange(full.size(0), device=full.device)
    full_logit = full[idx, pred]

    deltas = {}
    for name in model.branches:
        ablated = model(x, edge_index, disable_branch=name)
        drop = full_logit - ablated[idx, pred]
        deltas[name] = drop[mask] if mask is not None else drop
    return deltas


@torch.no_grad()
def gate_attribution(model, x, edge_index, mask=None) -> dict[str, torch.Tensor]:
    """Note: in residual fusion mode the trunk branch has no gate and is
    absent from the result (it is always fully present by construction)."""
    model.eval()
    model(x, edge_index)
    out = {}
    for name, g in model.fusion.gate_dict().items():
        out[name] = g[mask] if mask is not None else g
    return out


def h4_correlation(model, x, edge_index, mask=None) -> dict[str, float]:
    """Spearman rho between gate weight and leave-branch-out delta, per
    gated branch (trunk excluded in residual mode)."""
    lbo = leave_branch_out(model, x, edge_index, mask)
    gates = gate_attribution(model, x, edge_index, mask)
    return {
        name: float(spearmanr(gates[name].cpu().numpy(),
                              lbo[name].cpu().numpy()).statistic)
        for name in gates
    }
