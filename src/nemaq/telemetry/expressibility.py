"""Circuit diagnostics: expressibility (Sim et al. 2019) and
Meyer-Wallach entanglement capability.

Expressibility: KL divergence between the fidelity distribution of states
prepared with random parameter pairs and the Haar-random fidelity
distribution P_Haar(F) = (2^n - 1)(1 - F)^(2^n - 2). Lower KL = more
expressive. Reported per (n_qubits, depth) cell of the sweep.
"""
import numpy as np
import pennylane as qml


def _random_state(n_qubits: int, depth: int, rng: np.random.Generator) -> np.ndarray:
    dev = qml.device("default.qubit", wires=n_qubits)
    shape = qml.StronglyEntanglingLayers.shape(n_layers=depth, n_wires=n_qubits)
    weights = rng.uniform(0, 2 * np.pi, size=shape)
    inputs = rng.uniform(0, 2 * np.pi, size=n_qubits)

    @qml.qnode(dev)
    def circuit():
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return qml.state()

    return circuit()


def expressibility_kl(n_qubits: int, depth: int, n_pairs: int = 2000,
                      n_bins: int = 75, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    fids = np.empty(n_pairs)
    for i in range(n_pairs):
        s1 = _random_state(n_qubits, depth, rng)
        s2 = _random_state(n_qubits, depth, rng)
        fids[i] = np.abs(np.vdot(s1, s2)) ** 2

    edges = np.linspace(0, 1, n_bins + 1)
    hist, _ = np.histogram(fids, bins=edges)
    p = hist / hist.sum()

    d = 2 ** n_qubits
    centers = (edges[:-1] + edges[1:]) / 2
    haar = (d - 1) * (1 - centers) ** (d - 2)
    q = haar / haar.sum()

    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / np.maximum(q[mask], 1e-12))))


def meyer_wallach(n_qubits: int, depth: int, n_samples: int = 200,
                  seed: int = 0) -> float:
    """Q in [0, 1]; 0 = product states, 1 = maximal average entanglement."""
    rng = np.random.default_rng(seed)
    qs = []
    for _ in range(n_samples):
        state = _random_state(n_qubits, depth, rng)
        psi = state.reshape([2] * n_qubits)
        purity_sum = 0.0
        for k in range(n_qubits):
            m = np.moveaxis(psi, k, 0).reshape(2, -1)
            rho = m @ m.conj().T
            purity_sum += np.real(np.trace(rho @ rho))
        qs.append(2 * (1 - purity_sum / n_qubits))
    return float(np.mean(qs))
