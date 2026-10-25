"""
Solver calling the Marmote Value Iteration solver in C++.
"""

import numpy as np
from marmote.core import MarmoteInterval
from marmote.mdp import DiscountedMDP, SolutionMDP

from mdpforge.core.conversion import compute_marmote_args
from mdpforge.core.mdp import MDP
from mdpforge.core.solve_result import SolveResult
from mdpforge.core.solver import GenericSolver

METADATA = {
    "display_name": "VI · Marmote",
    "family": "vi",
    "backend": "marmote",
    "description": "Marmote C++ value iteration with standard Bellman sweeps.",
}


class Solver(GenericSolver):
    solver_type = "vi"

    def __init__(
        self,
        model: MDP,
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
        self.name = "VI Marmote"

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

        self.opt: SolutionMDP = self.mdp.ValueIteration(
            self.final_precision, self.max_iterations
        )

        return SolveResult(
            np.array([self.opt.getValueIndex(s) for s in range(self.model.state_dim)]),
            None,
        )
