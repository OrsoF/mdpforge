"""Focused Topological Value Iteration for positive-cost SSPs.

Certified action elimination: if r(s,a)+P_a U < L(s), action a is strictly
suboptimal for an undiscounted maximum-reward SSP (U upper, L lower bound).
Then apply SCC-ordered TVI to the reduced graph. This implements the core
FTVI pruning + topological phase, without the paper's full DFS trial scheduling.
"""

from __future__ import annotations

import numpy as np

from mdpforge.solvers.total._ssp_topology import (
    SSPBase,
    solve_topologically,
    validate_extra,
)

METADATA = {
    "display_name": "SSP Focused TVI · mdpforge",
    "family": "vi",
    "backend": "mdpforge",
    "description": "Safe bound-based action pruning followed by SCC-ordered SSP VI.",
}


class Solver(SSPBase):
    def __init__(
        self,
        model,
        final_precision=1e-3,
        *,
        search_sweeps=6,
        local_sweeps=1000,
        max_iterations=100_000,
    ):
        super().__init__(model, final_precision, max_iterations=max_iterations)
        validate_extra("search_sweeps", search_sweeps, minimum=0)
        validate_extra("local_sweeps", local_sweeps)
        self.search_sweeps = search_sweeps
        self.local_sweeps = local_sweeps
        self.n_pruned_actions = 0

    def run(self):
        p = self.problem
        upper = p.upper.copy()
        for _ in range(self.search_sweeps):
            q = p.q(upper)
            upper = np.minimum(upper, np.max(q, axis=1))
            p.n_backups += len(p.transient)
            p.improve_lower()
        optimistic_q = p.q(upper)
        # A numerical safety margin avoids accidentally deleting near-ties.
        margin = max(1e-11, self.final_precision * 0.01)
        allowed = optimistic_q >= p.lower[:, None] - margin
        for g in p.goals:
            allowed[g, :] = True
        if np.any(allowed.sum(axis=1) == 0):
            raise RuntimeError("FTVI: incorrect bounds left a state with no action")
        self.n_pruned_actions = int((~allowed[p.transient]).sum())
        answer = solve_topologically(
            p,
            upper,
            allowed=allowed,
            max_iterations=self.max_iterations,
            local_sweeps=self.local_sweeps,
        )
        return self.finish(answer)
