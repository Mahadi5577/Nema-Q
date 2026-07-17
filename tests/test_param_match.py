"""Design rule #3: parameter matching between PQC and surrogate is asserted.

If this test fails, the dequantization control is invalid and every
NEMA-Q vs NEMA-C comparison is meaningless.
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("pennylane")

from nemaq.models.quantum import QuantumBranch
from nemaq.models.surrogate import SurrogateBranch, pqc_param_count

TOLERANCE = 0.15  # surrogate core within 15% of PQC trainable circuit params


@pytest.mark.parametrize("nq,depth", [(4, 1), (4, 2), (6, 2), (8, 4)])
def test_surrogate_matches_pqc(nq, depth):
    q = QuantumBranch(in_dim=32, n_qubits=nq, depth=depth, out_dim=16)
    s = SurrogateBranch(in_dim=32, n_qubits=nq, depth=depth, out_dim=16)

    pqc_params = q.circuit_weights.numel()
    assert pqc_params == pqc_param_count(nq, depth)

    core_params = s.core_param_count()
    rel = abs(core_params - pqc_params) / pqc_params
    assert rel <= TOLERANCE, (
        f"surrogate core {core_params} vs PQC {pqc_params} ({rel:.0%} off)"
    )

    # identical wrapper layers (compress/project) => total diff == core diff
    q_compress = sum(p.numel() for p in q.compress.parameters())
    s_compress = sum(p.numel() for p in s.compress.parameters())
    assert q_compress == s_compress


def test_frozen_random_has_no_trainable_circuit():
    q = QuantumBranch(in_dim=32, n_qubits=4, depth=2, frozen_random=True)
    assert not q.circuit_weights.requires_grad
