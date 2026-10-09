"""
Solver calling the MDP Toolbox Modified Policy Iteration solver.
"""

import time

import numpy as np
from mdptoolbox.mdp import PolicyIterationModified

from mdpforge.core.model import MDPProtocol
from mdpforge.core.precision import certify_value


class Solver:
    def __init__(
        self,
        model: MDPProtocol,
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

        self.value = certify_value(
            self.model, np.asarray(self.mpi.V), self.gamma, self.epsilon
        )
        self.policy = np.array(self.mpi.policy)
        self.runtime = time.time() - start_time
