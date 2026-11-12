"""Total-reward occupancy-measure dual linear program using SciPy/HiGHS."""

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import eye, hstack

from mdpforge.core.solve_result import SolveResult

METADATA = {
    "display_name": "Total LP · SciPy · dual",
    "family": "lp",
    "backend": "scipy",
    "description": "Solve total-reward occupancy flows with SciPy/HiGHS.",
}


class Solver:
    """Solve the total-reward occupancy-measure linear program."""

    def __init__(self, model, *, method: str = "highs"):
        """method: 'highs', 'highs-ds', or 'highs-ipm'; HiGHS algorithm."""
        if method not in ("highs", "highs-ds", "highs-ipm"):
            raise ValueError("method must be 'highs', 'highs-ds' or 'highs-ipm'")
        self.model = model
        self.method = method
        self.name = __name__.rsplit(".", 1)[-1]

    def run(self):
        """Return values and a policy from transient-state occupancy flows."""
        n = self.model.state_dim
        terminal = set(getattr(self.model, "terminal_states", ()))
        if not terminal:
            raise ValueError("total-reward model must declare terminal_states")
        transient = np.array(
            [state for state in range(n) if state not in terminal],
            dtype=int,
        )
        identity = eye(len(transient), format="csr")
        constraints = hstack(
            [
                identity - transition[transient, :][:, transient].T
                for transition in self.model.transition_matrix
            ],
            format="csr",
        )
        rewards = self.model.reward_matrix[transient, :]
        result = linprog(
            -rewards.T.ravel(),
            A_eq=constraints,
            b_eq=np.ones(len(transient)),
            bounds=(0, None),
            method=self.method,
        )
        if not result.success:
            raise RuntimeError(
                f"{self.name} failed (status {result.status}): {result.message}"
            )
        value = np.zeros(n)
        value[transient] = -result.eqlin.marginals
        policy = np.zeros(n, dtype=int)
        occupancy = result.x.reshape(self.model.action_dim, len(transient)).T
        policy[transient] = occupancy.argmax(axis=1)
        return SolveResult(value, policy, validation_method="residual")
