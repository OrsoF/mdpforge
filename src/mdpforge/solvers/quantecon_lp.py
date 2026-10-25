"""QuantEcon dense simplex linear programming; transitions are densified locally."""

import numpy as np

from mdpforge.core.mdp import MDP
from mdpforge.core.solve_result import SolveResult
from mdpforge.core.solver import GenericSolver
from mdpforge.utils.quantecon import create_quantecon

METADATA = {
    "display_name": "LP · QuantEcon",
    "family": "lp",
    "backend": "quantecon",
    "description": (
        "QuantEcon dense simplex linear programming; transitions are densified locally."
    ),
}


class Solver(GenericSolver):
    def __init__(
        self,
        model: MDP,
        discount: float,
        *,
        max_iterations: int = 100_000,
    ):
        """Configure QuantEcon LP with native initialization.

        max_iterations: int > 0; simplex pivot budget; default 100_000.
        """
        assert 0 < discount < 1
        assert isinstance(max_iterations, int) and max_iterations > 0
        super().__init__(model, discount, None, max_iterations=max_iterations)
        self.name = "QuantEcon LP"

    def _solve(self) -> SolveResult:
        """Call QuantEcon's public solver with its native stopping rule."""
        problem = create_quantecon(self.model, self.discount, dense=True)
        result = problem.solve(
            method="linear_programming",
            max_iter=self.max_iterations,
        )
        return SolveResult(np.asarray(result.v), np.asarray(result.sigma, dtype=int))
