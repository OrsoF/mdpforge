"""Scalar-gain average-reward Bellman linear program using SciPy/HiGHS."""

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye, hstack, vstack

from mdpforge.solvers.average._common import relative_result

METADATA = {
    "display_name": "Average LP · SciPy · primal",
    "family": "lp",
    "backend": "scipy",
    "description": "Scalar-gain average-reward Bellman inequalities using SciPy/HiGHS.",
}


class Solver:
    """Solve the Bellman LP assuming the optimal average gain is state independent."""

    def __init__(self, model, *, method="highs"):
        """method: 'highs' (auto), 'highs-ds' (simplex), 'highs-ipm' (barrier)."""
        if method not in ("highs", "highs-ds", "highs-ipm"):
            raise ValueError("method must be 'highs', 'highs-ds' or 'highs-ipm'")
        self.model = model
        self.method = method
        self.name = __name__.rsplit(".", 1)[-1]

    def run(self):
        """Return the optimal scalar gain and a relative feasible Bellman potential."""
        n = self.model.state_dim
        identity = eye(n, format="csr")
        # Minimize g subject to g + h(s) >= r(s,a) + P[a]h, with h(0) = 0.
        potential_constraints = vstack(
            [transition - identity for transition in self.model.transition_matrix],
            format="csr",
        )
        gain_column = csr_matrix(-np.ones((n * self.model.action_dim, 1)))
        constraints = hstack([gain_column, potential_constraints], format="csr")
        objective = np.zeros(n + 1)
        objective[0] = 1.0
        result = linprog(
            objective,
            A_ub=constraints,
            b_ub=-self.model.reward_matrix.T.ravel(),
            bounds=[(None, None), (0.0, 0.0)] + [(None, None)] * (n - 1),
            method=self.method,
        )
        if not result.success:
            raise RuntimeError(
                f"{self.name} failed (status {result.status}): {result.message}"
            )
        # Feasible inequalities alone do not identify optimal transient actions.
        return relative_result(result.x[1:], None, gain=result.x[0])
