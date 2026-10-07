import time

import numpy as np

from core.model import SPARSE, GenericModel
from core.solver import GenericSolver
from utils.bellman import optimal_bellman_operator


class Solver(GenericSolver):
    def __init__(
        self,
        model: GenericModel,
        discount: float,
        final_precision: float = 1e-3,
        mode: str = SPARSE,
        initial_value: np.ndarray = None,
    ):
        self.model = model
        self.discount = discount
        self.epsilon = final_precision / 10
        self.mode = mode

        if initial_value is not None:
            self.value = initial_value
        else:
            self.value = np.zeros(self.model.state_dim)

        self.model._convert_model(self.mode)
        self.name = "VI"

    def run(self):
        start_time = time.time()
        tolerance = (
            0.1 * self.epsilon * (1 - self.discount)
            if self.discount < 1.0
            else 0.1 * self.epsilon
        )

        while True:
            new_value = optimal_bellman_operator(self.model, self.value, self.discount)
            bellman_residual = np.linalg.norm(new_value - self.value, ord=np.inf)
            if bellman_residual < tolerance:
                self.value = new_value
                break
            self.value = new_value

        self.runtime = time.time() - start_time
        self.policy = None
