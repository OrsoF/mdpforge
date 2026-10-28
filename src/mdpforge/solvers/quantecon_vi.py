"""QuantEcon sparse value iteration with its native norm-based stopping rule."""

import numpy as np

from mdpforge.core.mdp import MDP
from mdpforge.core.solve_result import SolveResult
from mdpforge.core.solver import GenericSolver
from mdpforge.utils.solver_backends import create_quantecon
from mdpforge.utils.solver_names import solver_names

METADATA = {
    "display_name": "VI · QuantEcon",
    "family": "vi",
    "backend": "quantecon",
    "description": (
        "QuantEcon sparse value iteration with its native norm-based stopping rule."
    ),
}


class Solver(GenericSolver):
    solver_type = "vi"

    def __init__(
        self,
        model: MDP,
        discount: float,
        final_precision: float = 1e-3,
        *,
        max_iterations: int = 100_000,
    ):
        """Configure QuantEcon VI with native initialization.

        final_precision: finite float > 0; absolute value error target; default 1e-3.
        max_iterations: int > 0; outer iteration budget; default 100_000.
        """
        assert 0 < discount < 1
        assert np.isfinite(final_precision) and final_precision > 0
        assert isinstance(max_iterations, int) and max_iterations > 0
        super().__init__(
            model, discount, final_precision, max_iterations=max_iterations
        )
        self.name, self.verbose_name = solver_names(Solver, locals())

    def _solve(self) -> SolveResult:
        """Call QuantEcon's public solver with its native stopping rule."""
        problem = create_quantecon(self.model, self.discount)
        # QuantEcon bounds the returned value error by epsilon / 2.
        result = problem.solve(
            method="value_iteration",
            epsilon=2 * self.final_precision,
            max_iter=self.max_iterations,
        )
        return SolveResult(np.asarray(result.v), np.asarray(result.sigma, dtype=int))
