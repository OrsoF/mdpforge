"""
Solver calling the Marmote Modified Policy Iteration solver in C++.
"""

from time import time

import numpy as np
from marmote.core import MarmoteInterval
from marmote.mdp import FeedbackSolutionMDP, TotalRewardMDP

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
        self.delta = final_precision
        self.max_policy_update = int(1e8)
        self.max_step_policy_evaluation = int(1e5)
        self.name = "PIMGSmarmote"

        self.model._model_to_marmote()

    def run(self):
        self.state_space = MarmoteInterval(0, int(self.model.state_dim - 1))
        self.action_space = MarmoteInterval(0, int(self.model.action_dim - 1))

        self.mdp = TotalRewardMDP(
            "max",
            self.state_space,
            self.action_space,
            self.model.transition_matrix,
            self.model.reward_matrix,
        )

        self.start_time = time()

        self.opt: FeedbackSolutionMDP = self.mdp.PolicyIterationModifiedGS(
            self.epsilon,
            self.max_policy_update,
            self.delta,
            self.max_step_policy_evaluation,
        )

        self.runtime = time() - self.start_time

        self.value = np.array(
            [self.opt.getValueIndex(ss) for ss in range(self.model.state_dim)]
        )
        self.policy = None
