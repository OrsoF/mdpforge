"""
Solver calling the Marmote Value Iteration Gauss-Seidel solver in C++.
"""

from time import time

import numpy as np
from marmote.core import MarmoteInterval
from marmote.mdp import DiscountedMDP, SolutionMDP

from mdpforge.core.conversion import compute_marmote_args
from mdpforge.core.model import MDPProtocol
from mdpforge.core.precision import certify_value
from mdpforge.core.solve_result import SolveResult
from mdpforge.core.solver import GenericSolver
from mdpforge.utils.solver_options import positive_integer

METADATA = {
    "display_name": "VI · Marmote · GS",
    "family": "vi",
    "backend": "marmote",
    "description": (
        "Marmote C++ value iteration with Gauss-Seidel Bellman sweeps and "
        "independent residual certification."
    ),
}


class Solver(GenericSolver):
    solver_type = "vi"

    def __init__(
        self,
        model: MDPProtocol,
        discount: float,
        final_precision: float = 1e-3,
        *,
        max_iterations: int = 100_000_000,
    ):
        """Configure this solver.

        final_precision: finite float > 0; absolute value error target; default 1e-3.
        max_iterations: int > 0; outer iteration budget; default 100_000_000.
        """
        super().__init__(
            model, discount, final_precision, max_iterations=max_iterations
        )
        self.model = model
        self.discount = discount
        self.name = "VIGS Marmote"

        assert final_precision > 0, "final_precision must be positive"
        self.epsilon = final_precision
        self.max_iterations = positive_integer(max_iterations, "max_iterations")

        self.transitions, self.rewards = compute_marmote_args(self.model)

    def _solve(self) -> SolveResult:
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

        start_time = time()

        self.opt: SolutionMDP = self.mdp.ValueIterationGS(
            self.epsilon * (1 - self.discount), self.max_iterations
        )

        value = np.array(
            [self.opt.getValueIndex(ss) for ss in range(self.model.state_dim)]
        )
        value = certify_value(self.model, value, self.discount, self.epsilon)
        return SolveResult(value, None, time() - start_time)
