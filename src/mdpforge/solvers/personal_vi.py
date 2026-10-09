import time

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
        initial_value: np.ndarray = None,
    ):
        assert 0 < discount < 1, "discount must be strictly between 0 and 1"
        self.model = model
        self.discount = discount
        assert final_precision > 0, "final_precision must be positive"
        self.epsilon = final_precision

        if initial_value is not None:
            self.value = initial_value
        else:
            self.value = np.zeros(self.model.state_dim)

        self.name = "VI"

    def run(self):
        start_time = time.time()
        tolerance = self.epsilon * (1 - self.discount)

        while True:
            new_value = optimal_bellman_operator(self.model, self.value, self.discount)
            bellman_residual = np.linalg.norm(new_value - self.value, ord=np.inf)
            if bellman_residual < tolerance:
                self.value = new_value
                break
            self.value = new_value

        self.runtime = time.time() - start_time
        self.policy = None
