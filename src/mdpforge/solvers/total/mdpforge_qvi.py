"""Q-value iteration for total-reward MDPs."""

import numpy as np

from mdpforge.core.operators import q_optimal_bellman_operator
from mdpforge.core.solve_result import SolveResult
from mdpforge.solvers.total._common import check_options

METADATA = {
    "display_name": "Total QVI · mdpforge",
    "family": "qvi",
    "backend": "mdpforge",
    "description": "Total-reward Q-value iteration with a Bellman-residual threshold.",
}


class Solver:
    """Iterate undiscounted state-action values."""

    def __init__(self, model, final_precision=1e-3, *, max_iterations=100_000):
        check_options(model, final_precision, max_iterations)
        self.model = model
        self.final_precision = final_precision
        self.max_iterations = max_iterations
        self.name = __name__.rsplit(".", 1)[-1]

    def run(self):
        """Return values and policy when the Q residual is small."""
        q_value = np.zeros((self.model.state_dim, self.model.action_dim))
        for _ in range(self.max_iterations):
            next_q = q_optimal_bellman_operator(self.model, q_value, 1.0)
            if not np.isfinite(next_q).all():
                raise RuntimeError("QVI stopped: non-finite values")
            if np.max(np.abs(next_q - q_value)) <= self.final_precision:
                return SolveResult(
                    next_q.max(axis=1),
                    next_q.argmax(axis=1),
                    validation_method="residual",
                )
            q_value = next_q
        raise RuntimeError("QVI stopped: max_iterations reached")
