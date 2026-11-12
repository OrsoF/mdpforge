"""Synchronous value iteration for total-reward MDPs."""

import numpy as np

from mdpforge.core.operators import optimal_bellman_operator
from mdpforge.core.solve_result import SolveResult
from mdpforge.solvers.total._common import check_options

METADATA = {
    "display_name": "Total VI · mdpforge",
    "family": "vi",
    "backend": "mdpforge",
    "description": "Total-reward value iteration with a Bellman-residual threshold.",
}


class Solver:
    """Iterate the undiscounted optimal Bellman operator."""

    def __init__(self, model, final_precision=1e-3, *, max_iterations=100_000):
        check_options(model, final_precision, max_iterations)
        self.model = model
        self.final_precision = final_precision
        self.max_iterations = max_iterations
        self.name = __name__.rsplit(".", 1)[-1]

    def run(self):
        """Return values when the full Bellman residual is small."""
        value = np.zeros(self.model.state_dim)
        for _ in range(self.max_iterations):
            next_value = optimal_bellman_operator(self.model, value, 1.0)
            if not np.isfinite(next_value).all():
                raise RuntimeError("VI stopped: non-finite values")
            if np.max(np.abs(next_value - value)) <= self.final_precision:
                return SolveResult(next_value, None, validation_method="residual")
            value = next_value
        raise RuntimeError("VI stopped: max_iterations reached")
