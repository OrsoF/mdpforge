"""Stationary occupancy-measure linear program using SciPy/HiGHS."""

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, eye, hstack, vstack

from mdpforge.solvers.average._common import relative_result

METADATA = {
    "display_name": "Average LP · SciPy · dual",
    "family": "lp",
    "backend": "linear",
}


class Solver:
    """Solve the occupancy LP for an MDP with a state-independent optimal gain."""

    def __init__(self, model, *, method: str = "highs"):
        """method: 'highs', 'highs-ds', or 'highs-ipm'; HiGHS algorithm."""
        if method not in ("highs", "highs-ds", "highs-ipm"):
            raise ValueError("method must be 'highs', 'highs-ds' or 'highs-ipm'")
        self.model = model
        self.method = method
        self.name = __name__.rsplit(".", 1)[-1]

    def run(self):
        """Return the scalar gain and a relative feasible Bellman potential."""
        n = self.model.state_dim
        rewards = self.model.reward_matrix.T.ravel()
        identity = eye(n, format="csr")
        # Neu, Jonsson and Gómez (2017), equations (1)–(2), in action-major order.
        flow = hstack(
            [identity - matrix.T for matrix in self.model.transition_matrix],
            format="csr",
        )
        mass = csr_matrix(np.ones((1, rewards.size)))
        rhs = np.zeros(n + 1)
        rhs[-1] = 1.0
        result = linprog(
            -rewards,
            A_eq=vstack([flow, mass], format="csr"),
            b_eq=rhs,
            bounds=(0, None),
            method=self.method,
        )
        if not result.success:
            raise RuntimeError(
                f"{self.name} failed (status {result.status}): {result.message}"
            )
        # SciPy minimizes the negated reward, so negate the flow multipliers.
        return relative_result(-result.eqlin.marginals[:n], None, gain=-result.fun)
