"""
Solver calling the Marmote Modified Policy Iteration solver in C++.
"""

import numpy as np
from marmote.core import MarmoteInterval
from marmote.mdp import DiscountedMDP, FeedbackSolutionMDP

from mdpforge.utils.marmote import compute_marmote_args
from mdpforge.core.mdp import MDP
from mdpforge.core.solve_result import SolveResult
from mdpforge.core.solver import GenericSolver

METADATA = {
    "display_name": "MPI · Marmote",
    "family": "mpi",
    "backend": "marmote",
    "description": (
        "Marmote C++ modified policy iteration with standard updates and at "
        "most ten inner policy-evaluation iterations."
    ),
}


class Solver(GenericSolver):
    def __init__(
        self,
        model: MDP,
        discount: float,
        final_precision: float = 1e-3,
        *,
        max_iterations: int = 100_000_000,
        max_evaluation_iterations: int = 10,
        evaluation_precision: float | None = None,
    ):
        """Configure this solver.

        final_precision: finite float > 0; absolute value error target; default 1e-3.
        max_iterations: int > 0; outer iteration budget; default 100_000_000.
        max_evaluation_iterations: int > 0; policy-evaluation iteration budget; default
            10.
        evaluation_precision: None for final_precision, or a finite float > 0; inner
            target; default None.
        """
        super().__init__(
            model, discount, final_precision, max_iterations=max_iterations
        )
        self.name = "MPI Marmote"

        self.evaluation_precision = (
            final_precision
            if evaluation_precision is None
            else float(evaluation_precision)
        )
        self.max_evaluation_iterations = max_evaluation_iterations

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

        self.opt: FeedbackSolutionMDP = self.mdp.PolicyIterationModified(
            self.final_precision,
            self.max_iterations,
            self.evaluation_precision,
            self.max_evaluation_iterations,
        )

        return SolveResult(
            np.array([self.opt.getValueIndex(s) for s in range(self.model.state_dim)]),
            None,
        )
