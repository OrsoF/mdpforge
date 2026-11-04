"""Discounted occupancy-measure dual linear program using SciPy/HiGHS."""

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import eye, hstack
from scipy.sparse.linalg import spsolve

from mdpforge.core.mdp import MDP
from mdpforge.core.operators import compute_transition_reward_policy
from mdpforge.core.solve_result import SolveResult
from mdpforge.core.solver import GenericSolver

METADATA = {
    "display_name": "LP · SciPy · dual",
    "family": "lp",
    "backend": "scipy",
    "description": (
        "Solve the discounted occupancy-measure dual linear program using "
        "SciPy/HiGHS, then recover and evaluate a deterministic policy."
    ),
}


class Solver(GenericSolver):
    def __init__(self, model: MDP, discount: float, *, method: str = "highs"):
        """method: 'highs' (automatic), 'highs-ds' (simplex), 'highs-ipm' (interior point)."""
        assert 0 < discount < 1
        if method not in ("highs", "highs-ds", "highs-ipm"):
            raise ValueError("method must be 'highs', 'highs-ds' or 'highs-ipm'")
        super().__init__(model, discount, None)
        self.method = method
        self.name = __name__.rsplit(".", 1)[-1]

    def _solve(self) -> SolveResult:
        identity = eye(self.model.state_dim, format="csr")
        constraints = hstack(
            [
                identity - self.discount * transition.T
                for transition in self.model.transition_matrix
            ],
            format="csr",
        )
        result = linprog(
            -self.model.reward_matrix.T.ravel(),
            A_eq=constraints,
            b_eq=np.ones(self.model.state_dim),
            bounds=(0, None),
            method=self.method,
        )
        if not result.success:
            raise RuntimeError(
                f"{self.name} failed (status {result.status}): {result.message}"
            )
        occupancy = result.x.reshape(self.model.action_dim, self.model.state_dim).T
        policy = occupancy.argmax(axis=1)
        transition, reward = compute_transition_reward_policy(self.model, policy)
        value = spsolve(identity - self.discount * transition, reward)
        return SolveResult(np.asarray(value), policy, validation_method="residual")
