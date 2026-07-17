"""Dequantization control (NEMA-C): classical branch parameter-matched to the PQC.

The headline quantum claim is NEMA-Q vs NEMA-C — same fusion, same head,
same training recipe; only this branch differs. Parameter matching is
asserted in tests/test_param_match.py, not assumed.
"""
import torch
from torch import nn


def pqc_param_count(n_qubits: int, depth: int) -> int:
    """StronglyEntanglingLayers: depth * n_qubits * 3 rotation angles."""
    return depth * n_qubits * 3


class SurrogateBranch(nn.Module):
    """Mirrors QuantumBranch interface: compress -> bounded nonlinear core -> project.

    The core is a small MLP whose hidden width is solved so its parameter
    count matches the PQC's trainable circuit weights as closely as possible
    (always within one hidden unit's worth; exact count reported).
    """

    def __init__(self, in_dim: int, n_qubits: int = 4, depth: int = 2, out_dim: int = 16):
        super().__init__()
        self.compress = nn.Linear(in_dim, n_qubits)  # identical to quantum branch
        target = pqc_param_count(n_qubits, depth)
        # core: Linear(n_qubits->h) + Linear(h->n_qubits), params = h*(2*n_qubits+1)+n_qubits
        h = max(1, round((target - n_qubits) / (2 * n_qubits + 1)))
        self.core = nn.Sequential(
            nn.Linear(n_qubits, h), nn.Tanh(), nn.Linear(h, n_qubits), nn.Tanh(),
        )
        self.project = nn.Linear(n_qubits, out_dim)  # identical to quantum branch

    def forward(self, x, edge_index=None):
        angles = torch.tanh(self.compress(x)) * torch.pi
        return self.project(self.core(angles))

    def core_param_count(self) -> int:
        return sum(p.numel() for p in self.core.parameters())
