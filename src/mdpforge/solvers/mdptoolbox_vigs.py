"""
Solver calling the MDP Toolbox Value Iteration solver.
"""

import time

import numpy as np
from mdptoolbox.mdp import ValueIterationGS

from mdpforge.core.model import MDP
from mdpforge.core.precision import certify_value
from mdpforge.core.solver import GenericSolver


class Solver(GenericSolver):
    solver_type = "vi"

    def __init__(
        self,
        model: MDP,
        discount: float,
        final_precision: float = 1e-3,
    ):
        assert 0 < discount < 1, "discount must be strictly between 0 and 1"
        self.model = model
        self.discount = discount
        assert final_precision > 0, "final_precision must be positive"
        self.epsilon = final_precision
        self.name = "VIGS MDPToolbox"
        self.max_iter = int(1e8)

        self.value: np.ndarray
        self.policy: np.ndarray

    def run(self):
        start_time = time.time()

        transitions = np.array([matrix.toarray() for matrix in self.model.transition_matrix])
        self.vi = ValueIterationGS(
            transitions,
            self.model.reward_matrix,
            discount=self.discount,
            epsilon=self.epsilon * (1 - self.discount),
            max_iter=self.max_iter,
        )
        # Toolbox's VI iteration bound is not a bound for Gauss-Seidel sweeps.
        self.vi.max_iter = self.max_iter
        self.vi.run()

        self.value = np.array(self.vi.V)
        self.policy = np.array(self.vi.policy)

        self.value = certify_value(self.model, self.value, self.discount, self.epsilon)
        self.runtime = time.time() - start_time
