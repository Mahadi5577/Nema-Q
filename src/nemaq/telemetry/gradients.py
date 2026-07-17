"""Per-branch gradient telemetry — the barren-plateau instrument (H3).

After each backward pass, records per-branch gradient L2 norm and
per-parameter gradient variance. For the quantum branch, circuit weights are
additionally tracked in isolation (``circuit_grad_var``): pooling them with
the large classical compress/project layers drowns the barren-plateau signal
in near-zero classical gradients.

H3 criterion (relative, not absolute): the circuit-weight gradient variance
must stay within ``H3_RELATIVE_ORDERS`` orders of magnitude of the best
classical branch's variance. Absolute floors from the BP literature assume
variance over random inits of circuit params only and do not transfer to a
trained mixed-device hybrid; the classical branches provide the matched
"healthy gradient" reference at the same loss scale.
"""
from collections import defaultdict

import torch
from torch import nn

H3_RELATIVE_ORDERS = 2.0  # circuit var may lag classical var by <= 2 orders of magnitude


class GradientTelemetry:
    def __init__(self, model: nn.Module, branch_attr: str = "branches"):
        self.model = model
        self.branch_attr = branch_attr
        self.history: dict[str, list[dict]] = defaultdict(list)

    def _branch_modules(self) -> dict[str, nn.Module]:
        branches = getattr(self.model, self.branch_attr, None)
        if branches is not None:
            return dict(branches.items())
        return {"model": self.model}

    @staticmethod
    def _grad_stats(params) -> dict | None:
        # .cpu(): branches may live on different devices (PQC is CPU-pinned
        # while classical branches sit on GPU); stats are scalars anyway.
        grads = [p.grad.flatten().cpu() for p in params
                 if p.grad is not None and p.requires_grad]
        if not grads:
            return None
        g = torch.cat(grads)
        return {
            "grad_norm": float(g.norm()),
            "grad_var": float(g.var(unbiased=False)),
            "grad_absmean": float(g.abs().mean()),
        }

    @torch.no_grad()
    def record(self, step: int) -> dict[str, dict]:
        """Call after loss.backward(), before optimizer.step()."""
        snapshot = {}
        for name, module in self._branch_modules().items():
            entry = self._grad_stats(module.parameters())
            if entry is None:
                continue
            entry["step"] = step
            # isolate PQC circuit weights (QuantumBranch exposes them)
            cw = getattr(module, "circuit_weights", None)
            if cw is not None and cw.grad is not None:
                entry["circuit_grad_var"] = float(
                    cw.grad.flatten().cpu().var(unbiased=False))
            self.history[name].append(entry)
            snapshot[name] = entry
        return snapshot

    def summary(self) -> dict:
        out = {}
        for name, entries in self.history.items():
            variances = [e["grad_var"] for e in entries]
            s = {
                "final_grad_var": variances[-1],
                "min_grad_var": min(variances),
                "n_steps": len(variances),
            }
            circuit = [e["circuit_grad_var"] for e in entries
                       if "circuit_grad_var" in e]
            if circuit:
                s["min_circuit_grad_var"] = min(circuit)
                s["final_circuit_grad_var"] = circuit[-1]
            out[name] = s

        # H3 verdict: circuit variance vs best classical-branch variance
        circuit_branches = {n: s for n, s in out.items()
                            if "min_circuit_grad_var" in s}
        classical = [s["min_grad_var"] for n, s in out.items()
                     if n not in circuit_branches]
        if circuit_branches and classical:
            ref = max(classical)
            for n, s in circuit_branches.items():
                ratio_ok = s["min_circuit_grad_var"] >= ref * 10 ** (-H3_RELATIVE_ORDERS)
                s["h3_pass_relative"] = bool(ratio_ok)
                s["h3_classical_reference_var"] = ref
        return out
