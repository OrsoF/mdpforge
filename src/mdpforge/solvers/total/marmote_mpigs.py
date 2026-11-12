"""Total-reward MPI · Marmote · GS through Marmote's public API."""

import numpy as np
from marmote.core import MarmoteInterval
from marmote.mdp import TotalRewardMDP

from mdpforge.core.solve_result import SolveResult
from mdpforge.solvers.total._common import check_options
from mdpforge.utils.solver_backends import compute_marmote_args

METADATA = {
    "display_name": "Total MPI · Marmote · GS",
    "family": "mpi",
    "backend": "marmote",
    "description": "Total-reward PolicyIterationModifiedGS using Marmote.",
}


class Solver:
    """Call Marmote's total-reward PolicyIterationModifiedGS method."""

    def __init__(
        self,
        model,
        final_precision=1e-3,
        *,
        max_iterations=100_000,
        max_evaluation_iterations=10,
        evaluation_precision=None,
    ):
        check_options(model, final_precision, max_iterations)
        if type(max_evaluation_iterations) is not int or max_evaluation_iterations < 1:
            raise ValueError("max_evaluation_iterations must be positive")
        if evaluation_precision is not None and (
            not np.isfinite(evaluation_precision) or evaluation_precision <= 0
        ):
            raise ValueError("evaluation_precision must be positive")
        self.max_evaluation_iterations = max_evaluation_iterations
        self.evaluation_precision = (
            final_precision if evaluation_precision is None else evaluation_precision
        )
        self.model = model
        self.final_precision = final_precision
        self.max_iterations = max_iterations
        self.name = __name__.rsplit(".", 1)[-1]

    def run(self):
        """Return Marmote's value and policy."""
        transitions, rewards = compute_marmote_args(self.model)
        states = MarmoteInterval(0, self.model.state_dim - 1)
        actions = MarmoteInterval(0, self.model.action_dim - 1)
        mdp = TotalRewardMDP("max", states, actions, transitions, rewards)
        solution = mdp.PolicyIterationModifiedGS(
            self.final_precision,
            self.max_iterations,
            self.evaluation_precision,
            self.max_evaluation_iterations,
        )
        return SolveResult(
            np.array([solution.getValueIndex(s) for s in range(self.model.state_dim)]),
            np.array([solution.getActionIndex(s) for s in range(self.model.state_dim)]),
            validation_method="residual",
        )
