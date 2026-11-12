"""Gauss-Seidel value iteration for total-reward MDPs."""

import numpy as np

from mdpforge.core.operators import optimal_bellman_operator
from mdpforge.core.solve_result import SolveResult
from mdpforge.solvers.total._common import check_options

METADATA = {
    "display_name": "Total GS VI · mdpforge",
    "family": "vi",
    "backend": "mdpforge",
    "description": (
        "Total-reward Gauss-Seidel sweeps with a Bellman-residual threshold."
    ),
}


class Solver:
    """Apply in-place undiscounted Bellman updates."""

    def __init__(self, model, final_precision=1e-3, *, max_iterations=100_000):
        check_options(model, final_precision, max_iterations)
        self.model = model
        self.final_precision = final_precision
        self.max_iterations = max_iterations
        self.name = __name__.rsplit(".", 1)[-1]

    def run(self):
        """Return values when the full Bellman residual is small."""
        value = np.zeros(self.model.state_dim)
        transitions = self.model.transition_matrix
        rewards = self.model.reward_matrix
        for _ in range(self.max_iterations):
            for state in range(self.model.state_dim):
                best = -np.inf
                for action, transition in enumerate(transitions):
                    lo, hi = transition.indptr[state : state + 2]
                    expected = np.dot(
                        transition.data[lo:hi], value[transition.indices[lo:hi]]
                    )
                    best = max(best, rewards[state, action] + expected)
                value[state] = best
            if not np.isfinite(value).all():
                raise RuntimeError("GS VI stopped: non-finite values")
            residual = optimal_bellman_operator(self.model, value, 1.0) - value
            if np.max(np.abs(residual)) <= self.final_precision:
                return SolveResult(value, None, validation_method="residual")
        raise RuntimeError("GS VI stopped: max_iterations reached")
