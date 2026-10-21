"""
Solver calling the Marmote Modified Policy Iteration solver in C++.
"""

from time import time

import numpy as np
from marmote.core import MarmoteInterval
from marmote.mdp import DiscountedMDP, FeedbackSolutionMDP

from mdpforge.core.conversion import compute_marmote_args
from mdpforge.core.model import MDPProtocol
from mdpforge.core.solve_result import SolveResult
from mdpforge.core.solver import GenericSolver
from mdpforge.utils.solver_options import positive_integer, positive_real

METADATA = {
    "display_name": "MPI · Marmote · GS",
    "family": "mpi",
    "backend": "marmote",
    "description": (
        "Marmote C++ modified policy iteration with Gauss-Seidel updates and "
        "at most ten inner policy-evaluation iterations."
    ),
}


class Solver(GenericSolver):
    def __init__(
        self,
        model: MDPProtocol,
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
        self.model = model
        self.discount = discount
        self.name = "MPIGS Marmote"

        self.epsilon_inner = (
            final_precision
            if evaluation_precision is None
            else positive_real(evaluation_precision, "evaluation_precision")
        )
        self.epsilon_outer = final_precision
        self.max_iterations = positive_integer(max_iterations, "max_iterations")
        self.max_evaluation_iterations = positive_integer(
            max_evaluation_iterations, "max_evaluation_iterations"
        )

        self.transitions, self.rewards = compute_marmote_args(self.model)

    def run(self) -> SolveResult:
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

        self.opt: FeedbackSolutionMDP = self.mdp.PolicyIterationModifiedGS(
            self.epsilon_outer,
            self.max_iterations,
            self.epsilon_inner,
            self.max_evaluation_iterations,
        )

        self.runtime = time() - self.start_time

        self.value = np.array(
            [self.opt.getValueIndex(ss) for ss in range(self.model.state_dim)]
        )
        self.policy = None
        return SolveResult.from_solver(self)
