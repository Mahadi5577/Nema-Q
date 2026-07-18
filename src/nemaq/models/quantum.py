"""Variational quantum branch (PennyLane TorchLayer).

Design constraints from the proposal (barren-plateau mitigation):
  - small and shallow: n_qubits <= 8, depth <= 4 (enforced),
  - angle embedding of a compressed feature vector,
  - StronglyEntanglingLayers ansatz,
  - per-qubit Pauli-Z expectations as output (local observables).

`frozen_random=True` gives the NEMA-R recipe control: same circuit, random
fixed parameters — separates "trainable quantum features" from "any fixed
nonlinear random projection".
"""
import pennylane as qml
import torch
from torch import nn

MAX_QUBITS = 8
MAX_DEPTH = 4


class QuantumBranch(nn.Module):
    def __init__(self, in_dim: int, n_qubits: int = 4, depth: int = 2,
                 out_dim: int = 16, frozen_random: bool = False,
                 shots: int | None = None, seed: int = 0,
                 angle_norm: str = "none"):
        super().__init__()
        assert n_qubits <= MAX_QUBITS, f"n_qubits > {MAX_QUBITS} violates BP budget"
        assert depth <= MAX_DEPTH, f"depth > {MAX_DEPTH} violates BP budget"
        assert angle_norm in ("none", "std"), f"angle_norm={angle_norm}"
        self.n_qubits = n_qubits
        self.depth = depth
        # "std": standardize the compressed pre-activations per dimension
        # before tanh, so the embedding uses the full +-pi angle range instead
        # of the near-identity regime that sparse features induce (the regime
        # the paper diagnoses as rationalizing H5). Post-freeze EXPLORATORY
        # knob — default "none" preserves the frozen behavior exactly.
        self.angle_norm = angle_norm

        # classical compressor: features -> rotation angles
        self.compress = nn.Linear(in_dim, n_qubits)

        dev = qml.device("default.qubit", wires=n_qubits, shots=shots)

        @qml.qnode(dev, interface="torch", diff_method="backprop" if shots is None else "parameter-shift")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

        shape = qml.StronglyEntanglingLayers.shape(n_layers=depth, n_wires=n_qubits)
        self.qlayer = qml.qnn.TorchLayer(circuit, {"weights": shape})

        if frozen_random:
            g = torch.Generator().manual_seed(seed)
            with torch.no_grad():
                self.qlayer.weights.copy_(
                    torch.rand(self.qlayer.weights.shape, generator=g) * 2 * torch.pi
                )
            self.qlayer.weights.requires_grad_(False)

        self.project = nn.Linear(n_qubits, out_dim)

    def angles(self, x) -> torch.Tensor:
        """Compressed features -> bounded rotation angles [N, n_qubits]."""
        pre = self.compress(x)
        if self.angle_norm == "std":
            pre = (pre - pre.mean(0)) / (pre.std(0) + 1e-8)
        return torch.tanh(pre) * torch.pi

    def forward(self, x, edge_index=None):
        angles = self.angles(x)
        # default.qubit simulates the statevector on CPU; keep the circuit
        # weights and inputs there even when the rest of the model is on GPU,
        # then return to the model device for fusion.
        out_device = angles.device
        if self.qlayer.weights.device.type != "cpu":
            self.qlayer.cpu()
        q = self.qlayer(angles.cpu())
        return self.project(q.to(out_device))

    @property
    def circuit_weights(self) -> torch.Tensor:
        return self.qlayer.weights
