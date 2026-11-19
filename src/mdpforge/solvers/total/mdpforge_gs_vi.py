"""Undiscounted Gauss-Seidel VI for positive-cost SSPs (not discounted)."""

from __future__ import annotations

from mdpforge.solvers.total._ssp_topology import SSPBase, validate_extra

METADATA = {
    "display_name": "SSP Gauss-Seidel VI · mdpforge",
    "family": "vi",
    "backend": "mdpforge",
    "description": "In-place VI at gamma=1 with a certified proper-policy/upper-bound gap.",
}


class Solver(SSPBase):
    def __init__(
        self,
        model,
        final_precision=1e-3,
        *,
        reverse_order=False,
        certify_every=5,
        max_iterations=100_000,
    ):
        super().__init__(model, final_precision, max_iterations=max_iterations)
        validate_extra("certify_every", certify_every)
        self.reverse_order = bool(reverse_order)
        self.certify_every = certify_every

    def run(self):
        p = self.problem
        upper = p.upper.copy()
        states = list(p.transient)
        if self.reverse_order:
            states.reverse()
        for iteration in range(1, self.max_iterations + 1):
            for s in states:
                upper[s] = min(upper[s], p.backup(int(s), upper))
                p.n_backups += 1
            if iteration % self.certify_every == 0 or iteration == 1:
                result = p.certify(upper, iterations=iteration)
                if result is not None:
                    return self.finish(result)
        raise RuntimeError("SSP Gauss-Seidel VI: max_iterations reached")
