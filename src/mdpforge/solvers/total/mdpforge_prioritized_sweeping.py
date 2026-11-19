"""Residual-based prioritized sweeping for undiscounted positive-cost SSPs.

This is generic prioritized sweeping, not McMahan & Gordon's IPS priority
with a value-dependent denominator. The priority queue and predecessors
reduce redundant state backups while global SSP gap certifies the result.
"""

from __future__ import annotations

import heapq

from mdpforge.solvers.total._ssp_topology import SSPBase, validate_extra

METADATA = {
    "display_name": "SSP Prioritized Sweeping · mdpforge",
    "family": "vi",
    "backend": "mdpforge",
    "description": "Residual-driven predecessor scheduling with rigorous SSP policy-gap termination.",
}


class Solver(SSPBase):
    def __init__(
        self, model, final_precision=1e-3, *, certify_every=0, max_iterations=100_000
    ):
        super().__init__(model, final_precision, max_iterations=max_iterations)
        validate_extra("certify_every", certify_every, minimum=0)
        self.certify_every = certify_every or max(5, model.state_dim)

    def run(self):
        p = self.problem
        upper = p.upper.copy()
        heap = []
        latest = {}
        counter = 0

        def schedule(s):
            nonlocal counter
            if s in p.goals:
                return
            residual = upper[s] - p.backup(s, upper)
            if residual > 1e-14:
                counter += 1
                latest[s] = counter
                heapq.heappush(heap, (-residual, counter, s))

        for s in p.transient:
            schedule(int(s))
        max_updates = self.max_iterations * max(1, len(p.transient))
        for update in range(1, max_updates + 1):
            if not heap:
                result = p.certify(upper, iterations=update)
                if result is not None:
                    return self.finish(result)
                for s in p.transient:
                    schedule(int(s))
                if not heap:
                    raise RuntimeError("SSP sweeping stalled: cannot certify bounds")
            while heap:
                _, marker, s = heapq.heappop(heap)
                if latest.get(s) == marker:
                    latest.pop(s)
                    break
            else:
                continue
            new = min(upper[s], p.backup(s, upper))
            if upper[s] - new > 1e-14:
                upper[s] = new
                p.n_backups += 1
                for parent in p.predecessors[s]:
                    schedule(parent)
            if update % self.certify_every == 0:
                result = p.certify(upper, iterations=update)
                if result is not None:
                    return self.finish(result)
        raise RuntimeError("SSP prioritized sweeping: max_iterations reached")
