"""Topological Value Iteration for undiscounted stochastic shortest paths.

Dai, Mausam, Weld and Goldsmith, JAIR 42 (2011). SCCs are resolved in
successor-first topological order, with local Gauss-Seidel updates.
"""

from __future__ import annotations

from mdpforge.solvers.total._ssp_topology import (
    SSPBase,
    solve_topologically,
    validate_extra,
)

METADATA = {
    "display_name": "SSP Topological VI · mdpforge",
    "family": "vi",
    "backend": "mdpforge",
    "description": "SCC-ordered value iteration for positive-cost undiscounted SSPs.",
}


class Solver(SSPBase):
    def __init__(
        self, model, final_precision=1e-3, *, local_sweeps=1000, max_iterations=100_000
    ):
        super().__init__(model, final_precision, max_iterations=max_iterations)
        validate_extra("local_sweeps", local_sweeps)
        self.local_sweeps = local_sweeps

    def run(self):
        answer = solve_topologically(
            self.problem,
            self.problem.upper.copy(),
            max_iterations=self.max_iterations,
            local_sweeps=self.local_sweeps,
        )
        return self.finish(answer)
