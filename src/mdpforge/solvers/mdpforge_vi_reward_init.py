import numpy as np

from mdpforge.core.model import MDPProtocol
from mdpforge.core.operators import optimal_bellman_operator
from mdpforge.core.solver import GenericSolver

METADATA = {
    "display_name": "VI · mdpforge · reward init",
    "family": "vi",
    "backend": "mdpforge",
    "description": (
        "Synchronous value iteration initialized from maximum immediate "
        "rewards divided by one minus the discount."
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
    ):
        super().__init__(model, discount, final_precision)
        self.initial_value = model.reward_matrix.max(axis=1) / (1 - discount)
        self.name = "mdpforge VI (reward init)"

    def _solve(self) -> np.ndarray:
        value = self.initial_value.copy()
        tolerance = self.final_precision * (1 - self.discount)

        while True:
            new_value = optimal_bellman_operator(self.model, value, self.discount)
            bellman_residual = np.linalg.norm(new_value - value, ord=np.inf)
            if bellman_residual < tolerance:
                return new_value
            value = new_value
