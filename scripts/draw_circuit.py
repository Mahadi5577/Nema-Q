"""Render the quantum branch's circuit (Fig. 2 of the QMI paper).

Gate-level view of the exact circuit in nemaq/models/quantum.py:
AngleEmbedding (R_Y) -> StronglyEntanglingLayers (depth 2) -> per-qubit <Z>.
Requires only pennylane + matplotlib (no torch).
"""
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pennylane as qml

N_QUBITS, DEPTH = 4, 2


def main(out="paper/latex/figs/quantum_circuit.pdf"):
    dev = qml.device("default.qubit", wires=N_QUBITS)

    @qml.qnode(dev)
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation="Y")
        qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
        return [qml.expval(qml.PauliZ(w)) for w in range(N_QUBITS)]

    shape = qml.StronglyEntanglingLayers.shape(n_layers=DEPTH, n_wires=N_QUBITS)
    weights = np.random.default_rng(0).uniform(0, 2 * np.pi, shape)
    fig, _ = qml.draw_mpl(circuit, level="device", style="black_white",
                          decimals=None)(np.zeros(N_QUBITS), weights)
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
