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
    ):
        """initial_value: None for zeros, or a finite NumPy vector of shape (S,)."""
        super().__init__(model, discount, final_precision)
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

        while True:
            new_value = optimal_bellman_operator(self.model, value, self.discount)
            bellman_residual = np.linalg.norm(new_value - value, ord=np.inf)
            if bellman_residual < tolerance:
                return new_value
            value = new_value
