"""Discounted value-function primal linear program using SciPy/HiGHS."""

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import eye, vstack

from mdpforge.core.mdp import MDP
from mdpforge.core.solve_result import SolveResult
from mdpforge.core.solver import GenericSolver

METADATA = {
    "display_name": "LP · SciPy · primal",
    "family": "lp",
    "backend": "scipy",
    "description": (
        "Solve the discounted value-function primal linear program with "
        "Bellman inequality constraints using SciPy/HiGHS."
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
        constraints = vstack(
            [
                self.discount * transition - identity
                for transition in self.model.transition_matrix
            ],
            format="csr",
        )
        result = linprog(
            np.ones(self.model.state_dim),
            A_ub=constraints,
            b_ub=-self.model.reward_matrix.T.ravel(),
            bounds=(None, None),
            method=self.method,
        )
        if not result.success:
            raise RuntimeError(
                f"{self.name} failed (status {result.status}): {result.message}"
            )
        return SolveResult(np.asarray(result.x), None, validation_method="residual")
