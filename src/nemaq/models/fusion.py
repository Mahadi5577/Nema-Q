"""Fusion of branch outputs. Two modes (config: model.fusion_mode):

- "residual" (default): the classical bypass is the trunk; every other branch
  enters as a sigmoid-gated residual whose gate bias starts at -2
  (sigma(-2) ~ 0.12), so training begins from "approximately the bypass GCN"
  and exotic branches must earn their gate mass. This enforces the proposal's
  graceful-degradation claim architecturally: worst case ~= plain GCN.

- "softmax": all branches compete through a softmax gate (zero-init = uniform
  start). Kept as an ablation — the Phase-4 matrix showed naive softmax
  competition is unstable (test acc 0.32–0.78 across seeds) regardless of
  branch type, classical surrogate included.

Gate values are exposed (`last_gates`) for telemetry and H4.
"""
import torch
from torch import nn

RESIDUAL_GATE_BIAS_INIT = -2.0


class GatedFusion(nn.Module):
    def __init__(self, branch_dims: dict[str, int], fused_dim: int,
                 mode: str = "residual", trunk: str = "bypass",
                 gate_bias_init: float = RESIDUAL_GATE_BIAS_INIT):
        super().__init__()
        if mode not in ("residual", "softmax"):
            raise ValueError(f"fusion_mode={mode}")
        if mode == "residual" and trunk not in branch_dims:
            raise ValueError(f"residual fusion needs trunk branch '{trunk}'")
        self.mode = mode
        self.trunk = trunk
        self.branch_names = list(branch_dims)
        self.residual_names = [n for n in self.branch_names if n != trunk]
        self.proj = nn.ModuleDict(
            {name: nn.Linear(d, fused_dim) for name, d in branch_dims.items()}
        )
        n_gated = len(self.residual_names) if mode == "residual" else len(self.branch_names)
        if n_gated > 0:
            self.gate = nn.Linear(fused_dim * len(branch_dims), n_gated)
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias,
                              gate_bias_init if mode == "residual" else 0.0)
        else:  # trunk-only ablation: nothing to gate
            self.gate = None
        self.last_gates: torch.Tensor | None = None  # [N, n_gated], telemetry

    def forward(self, branch_outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        projected = {n: self.proj[n](branch_outputs[n]) for n in self.branch_names}
        if self.gate is None:
            return projected[self.trunk]
        gate_in = torch.cat([projected[n] for n in self.branch_names], dim=-1)

        if self.mode == "softmax":
            gates = torch.softmax(self.gate(gate_in), dim=-1)      # [N, B]
            self.last_gates = gates.detach()
            stacked = torch.stack([projected[n] for n in self.branch_names], dim=1)
            return (gates.unsqueeze(-1) * stacked).sum(dim=1)

        gates = torch.sigmoid(self.gate(gate_in))                  # [N, B-1]
        self.last_gates = gates.detach()
        fused = projected[self.trunk]
        for i, name in enumerate(self.residual_names):
            fused = fused + gates[:, i:i + 1] * projected[name]
        return fused

    def gate_dict(self) -> dict[str, torch.Tensor]:
        """Map telemetry gate columns to branch names."""
        if self.gate is None or self.last_gates is None:
            return {}
        names = self.residual_names if self.mode == "residual" else self.branch_names
        return {n: self.last_gates[:, i] for i, n in enumerate(names)}
