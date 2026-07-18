"""NEMA-Q assembly + model factory.

All ablations are config swaps (design rule #2):
  geometry:     "hyperbolic" | "euclidean"          (H1 control)
  quantum_mode: "pqc" | "surrogate" | "frozen_random" | "off"  (H2 controls)
"""
import torch
from torch import nn

from .classical import GAT, GCN, MLP, GCNEncoder
from .fusion import GatedFusion
from .hyperbolic import HGCN, HyperbolicEncoder
from .quantum import QuantumBranch
from .surrogate import SurrogateBranch


class NemaQ(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, cfg: dict):
        super().__init__()
        hid = cfg.get("hidden", 64)
        emb = cfg.get("branch_dim", 16)
        fused = cfg.get("fused_dim", 32)
        dropout = cfg.get("dropout", 0.5)
        geometry = cfg.get("geometry", "hyperbolic")
        qmode = cfg.get("quantum_mode", "pqc")

        self.branches = nn.ModuleDict()
        if geometry == "hyperbolic":
            self.branches["geo"] = HyperbolicEncoder(
                in_dim, hid, emb, dropout,
                c=cfg.get("curvature", 1.0),
                learnable_c=cfg.get("learnable_curvature", True),
            )
        elif geometry == "euclidean":
            self.branches["geo"] = GCNEncoder(in_dim, hid, emb, dropout)
        elif geometry != "off":  # "off": trunk-only diagnostic ablation
            raise ValueError(f"geometry={geometry}")

        nq, depth = cfg.get("n_qubits", 4), cfg.get("q_depth", 2)
        anorm = cfg.get("angle_norm", "none")
        if qmode == "pqc":
            self.branches["q"] = QuantumBranch(in_dim, nq, depth, emb,
                                               shots=cfg.get("shots"),
                                               angle_norm=anorm)
        elif qmode == "frozen_random":
            self.branches["q"] = QuantumBranch(in_dim, nq, depth, emb,
                                               frozen_random=True,
                                               seed=cfg.get("q_seed", 0),
                                               angle_norm=anorm)
        elif qmode == "surrogate":
            self.branches["q"] = SurrogateBranch(in_dim, nq, depth, emb)
        elif qmode != "off":
            raise ValueError(f"quantum_mode={qmode}")

        self.branches["bypass"] = GCNEncoder(in_dim, hid, emb, dropout)

        self.fusion = GatedFusion({n: emb for n in self.branches}, fused,
                                  mode=cfg.get("fusion_mode", "residual"),
                                  gate_bias_init=cfg.get("gate_bias_init", -2.0))
        self.head = nn.Linear(fused, num_classes)
        # Per-branch auxiliary classifiers (deep supervision): each branch
        # learns discriminative features regardless of gate dynamics —
        # prevents the gate-collapse branch starvation seen with pure
        # softmax fusion.
        self.aux_heads = nn.ModuleDict(
            {n: nn.Linear(emb, num_classes) for n in self.branches}
        )
        self.last_aux_logits: dict[str, torch.Tensor] = {}
        self.last_branch_outputs: dict[str, torch.Tensor] = {}  # telemetry

    def forward(self, x, edge_index, disable_branch: str | None = None,
                branch_override: dict[str, torch.Tensor] | None = None):
        """`disable_branch` zeroes one branch output (leave-branch-out);
        `branch_override` substitutes a branch's output tensor (used by QOA
        to differentiate through the observable vector)."""
        outs = {}
        for name, branch in self.branches.items():
            if branch_override and name in branch_override:
                o = branch_override[name]
            else:
                o = branch(x, edge_index)
            if name == disable_branch:
                o = torch.zeros_like(o)
            outs[name] = o
        self.last_branch_outputs = {k: v.detach() for k, v in outs.items()}
        self.last_aux_logits = {n: self.aux_heads[n](outs[n]) for n in self.branches}
        return self.head(self.fusion(outs))


def build_model(name: str, in_dim: int, num_classes: int, cfg: dict) -> nn.Module:
    name = name.lower()
    hid = cfg.get("hidden", 64)
    dropout = cfg.get("dropout", 0.5)
    if name == "gcn":
        return GCN(in_dim, hid, num_classes, dropout)
    if name == "gat":
        return GAT(in_dim, cfg.get("gat_hidden", 8), num_classes,
                   cfg.get("heads", 8), cfg.get("gat_dropout", 0.6))
    if name == "mlp":
        return MLP(in_dim, hid, num_classes, dropout)
    if name == "hgcn":
        return HGCN(in_dim, hid, num_classes, dropout,
                    c=cfg.get("curvature", 1.0),
                    learnable_c=cfg.get("learnable_curvature", True))
    if name == "nemaq":
        return NemaQ(in_dim, num_classes, cfg)
    raise ValueError(f"Unknown model: {name}")
