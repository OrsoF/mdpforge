"""
Solver calling the Marmote Modified Policy Iteration solver in C++.
"""

from time import time

import numpy as np
from marmote.core import MarmoteInterval
from marmote.mdp import DiscountedMDP, FeedbackSolutionMDP

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
        self.name = "MPIGS Marmote"

        self.epsilon_inner = final_precision
        self.epsilon_outer = final_precision
        self.max_step_policy_update = int(1e8)
        self.max_step_policy_evaluation = 10

        self.model._model_to_marmote()

    def run(self):
        self.state_space = MarmoteInterval(0, int(self.model.state_dim - 1))
        self.action_space = MarmoteInterval(0, int(self.model.action_dim - 1))

        self.mdp = DiscountedMDP(
            "max",
            self.state_space,
            self.action_space,
            self.model.transition_matrix,
            self.model.reward_matrix,
            self.discount,
        )

        self.start_time = time()

        self.opt: FeedbackSolutionMDP = self.mdp.PolicyIterationModifiedGS(
            self.epsilon_outer,
            self.max_step_policy_update,
            self.epsilon_inner,
            self.max_step_policy_evaluation,
        )

        self.runtime = time() - self.start_time

        self.value = np.array(
            [self.opt.getValueIndex(ss) for ss in range(self.model.state_dim)]
        )
        self.policy = None
