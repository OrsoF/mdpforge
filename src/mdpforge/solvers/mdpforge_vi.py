import numpy as np

from mdpforge.core.mdp import MDP
from mdpforge.core.operators import optimal_bellman_operator
from mdpforge.core.solve_result import SolveResult
from mdpforge.core.solver import GenericSolver
from mdpforge.utils.solver_options import initial_vectors

METADATA = {
    "display_name": "VI · mdpforge",
    "family": "vi",
    "backend": "mdpforge",
    "description": (
        "Synchronous value iteration with a full Bellman-residual stopping "
        "threshold for absolute value precision."
    ),
}


class Solver(GenericSolver):
    solver_type = "vi"

    def __init__(
        self,
        model: MDP,
        discount: float,
        final_precision: float = 1e-3,
        initial_value: np.ndarray | None = None,
        *,
        max_iterations: int = 100_000,
    ):
        """Configure value iteration.

        initial_value: None for zeros, or a finite NumPy vector of shape (S,).
        max_iterations: int > 0, default 100_000, maximum Bellman updates.
        """
        super().__init__(
            model, discount, final_precision, max_iterations=max_iterations
        )
        value, _ = initial_vectors(model, initial_value)
        self.initial_value = np.zeros(model.state_dim) if value is None else value
        self.name = "mdpforge VI"

    def _solve(self) -> SolveResult:
        value = self.initial_value.copy()
        tolerance = self.final_precision * (1 - self.discount)

        for iteration in range(1, self.max_iterations + 1):
            new_value = optimal_bellman_operator(self.model, value, self.discount)
            bellman_residual = np.linalg.norm(new_value - value, ord=np.inf)
            if not np.isfinite(bellman_residual) or not np.isfinite(new_value).all():
                raise RuntimeError("VI stopped: non_finite values")
            if bellman_residual < tolerance:
                return SolveResult(
                    np.asarray(new_value),
                    None,
                    iteration,
                    "converged",
                )
            if np.array_equal(new_value, value):
                raise RuntimeError("VI stopped: stagnation")
            value = new_value
        raise RuntimeError("VI stopped: max_iterations reached")
