"""
Solver calling the Marmote Value Iteration Gauss-Seidel solver in C++.
"""

from time import time

import numpy as np
from marmote.core import MarmoteInterval
from marmote.mdp import DiscountedMDP, SolutionMDP

from mdpforge.core.model import GenericModel
from mdpforge.core.solver import GenericSolver


class Solver(GenericSolver):
    def __init__(
        self,
        model: GenericModel,
        discount: float,
        final_precision: float,
    ):
        self.model = model
        self.discount = discount
        self.name = "VIGS Marmote"

        self.epsilon = 1e-3
        self.max_iter = int(1e8)

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

        self.opt: SolutionMDP = self.mdp.ValueIterationGS(self.epsilon, self.max_iter)

        self.runtime = time() - self.start_time
        self.value = np.array(
            [self.opt.getValueIndex(ss) for ss in range(self.model.state_dim)]
        )
        self.policy = None
