"""
Solver calling the MDP Toolbox Modified Policy Iteration solver.
"""

import time

import numpy as np
from mdptoolbox.mdp import PolicyIterationModified

from core.model import GenericModel


class Solver:
    def __init__(
        self,
        model: GenericModel,
        discount: float,
        final_precision: float,
    ):
        self.model = model
        self.gamma = discount
        self.epsilon = final_precision
        self.max_step_policy_evaluation = 10

        self.name = "MPI MDPToolbox"
        self.value: np.ndarray
        self.policy: np.ndarray

    def run(self):
        start_time = time.time()

        self.pim = PolicyIterationModified(
            self.model.transition_matrix,
            self.model.reward_matrix,
            discount=self.gamma,
            epsilon=self.epsilon,
            max_iter=self.max_step_policy_evaluation,
            skip_check=True,
        )
        self.pim.run()
        self.runtime = time.time() - start_time

        self.value = np.array(self.pim.V)
        self.policy = np.array(self.pim.policy)
