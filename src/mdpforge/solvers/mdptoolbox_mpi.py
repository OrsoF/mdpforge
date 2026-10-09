"""
Solver calling the MDP Toolbox Modified Policy Iteration solver.
"""

import time

import numpy as np
from mdptoolbox.mdp import PolicyIterationModified

from mdpforge.core.model import MDP


class Solver:
    def __init__(
        self,
        model: MDP,
        discount: float,
        final_precision: float,
    ):
        assert 0 < discount < 1, "discount must be strictly between 0 and 1"
        self.model = model
        self.gamma = discount
        self.epsilon = final_precision
        self.max_step_policy_evaluation = 10

        self.name = "MPI MDPToolbox"
        self.value: np.ndarray
        self.policy: np.ndarray

    def run(self):
        start_time = time.time()

        self.mpi = PolicyIterationModified(
            self.model.transition_matrix,
            self.model.reward_matrix,
            discount=self.gamma,
            epsilon=self.epsilon,
            max_iter=self.max_step_policy_evaluation,
        )
        self.mpi.run()
        self.runtime = time.time() - start_time

        self.value = np.array(self.mpi.V)
        self.policy = np.array(self.mpi.policy)
