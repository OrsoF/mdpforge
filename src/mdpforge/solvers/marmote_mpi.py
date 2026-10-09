"""
Solver calling the Marmote Modified Policy Iteration solver in C++.
"""

from time import time

import numpy as np
from marmote.core import MarmoteInterval
from marmote.mdp import DiscountedMDP, FeedbackSolutionMDP

from mdpforge.core.conversion import compute_marmote_args
from mdpforge.core.model import MDP
from mdpforge.core.solver import GenericSolver


class Solver(GenericSolver):
    def __init__(
        self,
        model: MDP,
        discount: float,
        final_precision: float,
    ):
        assert 0 < discount < 1, "discount must be strictly between 0 and 1"
        self.model = model
        self.discount = discount
        self.name = "MPI Marmote"

        self.epsilon_inner = final_precision
        self.epsilon_outer = final_precision
        self.max_step_policy_update = int(1e8)
        self.max_step_policy_evaluation = 10

        self.transitions, self.rewards = compute_marmote_args(self.model)

    def run(self):
        self.state_space = MarmoteInterval(0, int(self.model.state_dim - 1))
        self.action_space = MarmoteInterval(0, int(self.model.action_dim - 1))

        self.mdp = DiscountedMDP(
            "max",
            self.state_space,
            self.action_space,
            self.transitions,
            self.rewards,
            self.discount,
        )

        self.start_time = time()

        self.opt: FeedbackSolutionMDP = self.mdp.PolicyIterationModified(
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
