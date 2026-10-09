"""
Solver calling the Marmote Value Iteration solver in C++.
"""

from time import time

import numpy as np
from marmote.core import MarmoteInterval
from marmote.mdp import DiscountedMDP, SolutionMDP

from mdpforge.core.conversion import compute_marmote_args
from mdpforge.core.model import MDPProtocol
from mdpforge.core.precision import certify_value
from mdpforge.core.solver import GenericSolver


class Solver(GenericSolver):
    solver_type = "vi"

    def __init__(
        self,
        model: MDPProtocol,
        discount: float,
        final_precision: float = 1e-3,
    ):
        assert 0 < discount < 1, "discount must be strictly between 0 and 1"
        self.model = model
        self.discount = discount
        self.name = "VI Marmote"

        assert final_precision > 0, "final_precision must be positive"
        self.epsilon = final_precision
        self.max_iter = int(1e8)

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

        self.opt: SolutionMDP = self.mdp.ValueIteration(
            self.epsilon * (1 - self.discount), self.max_iter
        )

        self.value = np.array(
            [self.opt.getValueIndex(ss) for ss in range(self.model.state_dim)]
        )
        self.value = certify_value(self.model, self.value, self.discount, self.epsilon)
        self.policy = None
        self.runtime = time() - self.start_time
