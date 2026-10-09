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
        initial_value: np.ndarray | None = None,
    ):
        super().__init__(model, discount, final_precision)

        if initial_value is not None:
            self.value = initial_value
        else:
            self.value = np.zeros(self.model.state_dim)

        self.name = "VI"

    def _solve(self) -> np.ndarray:
        tolerance = self.final_precision * (1 - self.discount)

        while True:
            new_value = optimal_bellman_operator(self.model, self.value, self.discount)
            bellman_residual = np.linalg.norm(new_value - self.value, ord=np.inf)
            if bellman_residual < tolerance:
                return new_value
            self.value = new_value
