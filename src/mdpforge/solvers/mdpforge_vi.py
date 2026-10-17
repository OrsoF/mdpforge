import numpy as np

from mdpforge.core.model import MDPProtocol
from mdpforge.core.operators import optimal_bellman_operator
from mdpforge.core.solver import GenericSolver

METADATA = {
    "display_name": "VI · mdpforge",
    "family": "vi",
    "backend": "mdpforge",
    "description": (
        "Synchronous value iteration with a full Bellman-residual stopping "
        "threshold for absolute value precision."
    ),
    "update_scheme": "synchronous",
}


class Solver(GenericSolver):
    solver_type = "vi"

    def __init__(
        self,
        model: MDPProtocol,
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
        self.initial_value = (
            np.zeros(model.state_dim)
            if initial_value is None
            else np.array(initial_value, dtype=float, copy=True)
        )
        if self.initial_value.shape != (model.state_dim,) or not np.all(
            np.isfinite(self.initial_value)
        ):
            raise ValueError("initial_value must contain one finite value per state")
        self.name = "mdpforge VI"

    def _solve(self) -> np.ndarray:
        value = self.initial_value.copy()
        tolerance = self.final_precision * (1 - self.discount)

        for iteration in range(1, self.max_iterations + 1):
            new_value = optimal_bellman_operator(self.model, value, self.discount)
            bellman_residual = np.linalg.norm(new_value - value, ord=np.inf)
            if self._check_iteration(
                iteration,
                bellman_residual,
                tolerance,
                new_value,
                stagnated=np.array_equal(new_value, value),
            ):
                return new_value
            value = new_value
