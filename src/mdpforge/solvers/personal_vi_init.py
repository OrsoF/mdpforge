import numpy as np

from mdpforge.core.model import MDPProtocol
from mdpforge.core.operators import optimal_bellman_operator
from mdpforge.core.solver import GenericSolver


class Solver(GenericSolver):
    solver_type = "vi"

    def __init__(
        self,
        model: MDPProtocol,
        discount: float,
        final_precision: float = 1e-3,
    ):
        super().__init__(model, discount, final_precision)

        self.value = model.reward_matrix.max(axis=1) / (1 - discount)

        self.name = "VI"

    def _solve(self) -> np.ndarray:
        tolerance = self.final_precision * (1 - self.discount)

        while True:
            new_value = optimal_bellman_operator(self.model, self.value, self.discount)
            bellman_residual = np.linalg.norm(new_value - self.value, ord=np.inf)
            if bellman_residual < tolerance:
                return new_value
            self.value = new_value
