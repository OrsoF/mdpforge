"""LAO*-style explicit greedy-envelope search for positive-cost SSPs.

An initial-state rooted partial-policy graph is expanded, and in-place
Bellman backups run only on visited states. Unvisited states are eventually
seeded for full-vector MDPForge output. This is an LAO*-style adaptation,
NOT a full implementation of the paper's Improved LAO* (ILAO*). Termination
is checked using the global certified lower/upper SSP policy gap.
"""

from __future__ import annotations

import numpy as np

from mdpforge.solvers.total._ssp_topology import SSPBase, validate_extra

METADATA = {
    "display_name": "SSP LAO*-style search · mdpforge",
    "family": "experimental",
    "backend": "mdpforge",
    "description": "LAO*-style greedy-envelope expansion and restricted VI with a full-state SSP gap certificate.",
}


class Solver(SSPBase):
    def __init__(
        self,
        model,
        final_precision=1e-3,
        *,
        initial_state=0,
        expansion_batch=32,
        certify_every=5,
        max_iterations=100_000,
    ):
        super().__init__(model, final_precision, max_iterations=max_iterations)
        validate_extra("initial_state", initial_state, minimum=0)
        if initial_state >= model.state_dim:
            raise ValueError("initial_state outside state space")
        validate_extra("expansion_batch", expansion_batch)
        validate_extra("certify_every", certify_every)
        self.initial_state = initial_state
        self.expansion_batch = expansion_batch
        self.certify_every = certify_every
        self.n_expanded = 0

    def run(self):
        p = self.problem
        upper = p.upper.copy()
        expanded = np.zeros(p.n, dtype=bool)
        expanded[list(p.goals)] = True
        for it in range(1, self.max_iterations + 1):
            # Build the current greedy envelope from a root, then next unseen roots.
            policy = p.greedy(upper)
            root = (
                self.initial_state
                if it == 1
                else (
                    int(np.flatnonzero(~expanded)[0])
                    if not expanded.all()
                    else self.initial_state
                )
            )
            stack = [root]
            seen = set()
            added = 0
            while stack and added < self.expansion_batch:
                s = int(stack.pop())
                if s in seen:
                    continue
                seen.add(s)
                if not expanded[s]:
                    expanded[s] = True
                    added += 1
                if s in p.goals:
                    continue
                successors, _ = p.succ[s][int(policy[s])]
                stack.extend(int(t) for t in successors if int(t) not in seen)
            # Dynamic programming restricted to explicitly expanded states.
            for s in np.flatnonzero(expanded):
                if s in p.goals:
                    continue
                upper[s] = min(upper[s], p.backup(int(s), upper))
                p.n_backups += 1
            self.n_expanded = int(expanded.sum())
            if expanded.all() and (it % self.certify_every == 0 or it == 1):
                result = p.certify(upper, iterations=it)
                if result is not None:
                    return self.finish(result)
        raise RuntimeError("SSP LAO*-style search: max_iterations reached")
