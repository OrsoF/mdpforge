"""
Solver calling the MDP Toolbox Value Iteration solver.
"""

import time

import numpy as np
from mdptoolbox.mdp import PolicyIteration

from core.model import GenericModel
from core.solver import GenericSolver


class Solver(GenericSolver):
    def __init__(
        self,
        model: GenericModel,
        discount: float,
        final_precision: float,
    ):
        self.model = model
        self.discount = discount
        self.epsilon = final_precision
        self.name = "VImdptoolbox"
        self.max_iter = int(1e8)

        self.value: np.ndarray
        self.policy: np.ndarray

    def run(self):
        start_time = time.time()

        self.vi = PolicyIteration(
            self.model.transition_matrix,
            self.model.reward_matrix,
            discount=self.discount,
            max_iter=self.max_iter,
            skip_check=True,
        )
        self.vi.run()
        self.runtime = time.time() - start_time

        self.value = np.array(self.vi.V)
        self.policy = np.array(self.vi.policy)
