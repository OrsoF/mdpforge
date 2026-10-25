"""QuantEcon sparse modified policy iteration with span stopping and value shift."""

import numpy as np

from mdpforge.core.mdp import MDP
from mdpforge.core.solve_result import SolveResult
from mdpforge.core.solver import GenericSolver
from mdpforge.utils.quantecon import create_quantecon

METADATA = {
    "display_name": "MPI · QuantEcon",
    "family": "mpi",
    "backend": "quantecon",
    "description": (
        "QuantEcon sparse modified policy iteration with span stopping and value shift."
    ),
}


class Solver(GenericSolver):
    def __init__(
        self,
        model: MDP,
        discount: float,
        final_precision: float = 1e-3,
        *,
        max_iterations: int = 100_000,
        max_evaluation_iterations: int = 20,
    ):
        """Configure QuantEcon MPI with native initialization.

        final_precision: finite float > 0; absolute value error target; default 1e-3.
        max_iterations: int > 0; outer iteration budget; default 100_000.
        max_evaluation_iterations: int > 0; partial evaluation steps; default 20.
        """
        assert 0 < discount < 1
        assert np.isfinite(final_precision) and final_precision > 0
        assert isinstance(max_iterations, int) and max_iterations > 0
        assert isinstance(max_evaluation_iterations, int)
        assert max_evaluation_iterations > 0
        super().__init__(
            model, discount, final_precision, max_iterations=max_iterations
        )
        self.max_evaluation_iterations = max_evaluation_iterations
        self.name = "QuantEcon MPI"

    def _solve(self) -> SolveResult:
        """Call QuantEcon's public solver with its native stopping rule."""
        problem = create_quantecon(self.model, self.discount)
        # QuantEcon bounds the returned value error by epsilon / 2.
        result = problem.solve(
            method="modified_policy_iteration",
            epsilon=2 * self.final_precision,
            max_iter=self.max_iterations,
            k=self.max_evaluation_iterations,
        )
        return SolveResult(np.asarray(result.v), np.asarray(result.sigma, dtype=int))
