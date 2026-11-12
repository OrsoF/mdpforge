"""Total-reward value-function primal linear program using SciPy/HiGHS."""

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import eye, vstack

from mdpforge.core.solve_result import SolveResult

METADATA = {
    "display_name": "Total LP · SciPy · primal",
    "family": "lp",
    "backend": "scipy",
    "description": "Solve total-reward Bellman inequalities with SciPy/HiGHS.",
}


class Solver:
    """Solve the total-reward value-function linear program."""

    def __init__(self, model, *, method: str = "highs"):
        """method: 'highs', 'highs-ds', or 'highs-ipm'; HiGHS algorithm."""
        if method not in ("highs", "highs-ds", "highs-ipm"):
            raise ValueError("method must be 'highs', 'highs-ds' or 'highs-ipm'")
        self.model = model
        self.method = method
        self.name = __name__.rsplit(".", 1)[-1]

    def run(self):
        """Return the optimal value vector with zero terminal values."""
        n = self.model.state_dim
        terminal = set(getattr(self.model, "terminal_states", ()))
        if not terminal:
            raise ValueError("total-reward model must declare terminal_states")
        identity = eye(n, format="csr")
        constraints = vstack(
            [transition - identity for transition in self.model.transition_matrix],
            format="csr",
        )
        bounds = [(0, 0) if state in terminal else (None, None) for state in range(n)]
        result = linprog(
            np.ones(n),
            A_ub=constraints,
            b_ub=-self.model.reward_matrix.T.ravel(),
            bounds=bounds,
            method=self.method,
        )
        if not result.success:
            raise RuntimeError(
                f"{self.name} failed (status {result.status}): {result.message}"
            )
        return SolveResult(np.asarray(result.x), None, validation_method="residual")
