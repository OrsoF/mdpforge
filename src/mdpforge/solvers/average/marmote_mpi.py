"""Average-reward modified policy iteration through Marmote's public API."""

import numpy as np
from marmote.core import MarmoteInterval
from marmote.mdp import AverageMDP

from mdpforge.solvers.average._common import (
    check_positive_integer,
    check_precision,
    relative_result,
)
from mdpforge.utils.solver_backends import compute_marmote_args

METADATA = {
    "display_name": "Average MPI · Marmote",
    "family": "mpi",
    "backend": "marmote",
    "description": (
        "Marmote modified policy iteration for unichain average-reward MDPs."
    ),
}


class Solver:
    """Run AverageMDP.PolicyIterationModified with native partial evaluation."""

    def __init__(
        self,
        model,
        final_precision=1e-3,
        *,
        max_iterations=100_000,
        max_evaluation_iterations=10,
        evaluation_precision=None,
    ):
        """Configure native MPI.

        final_precision: finite float > 0, outer convergence tolerance.
        max_iterations: int > 0, outer iteration budget.
        max_evaluation_iterations: int > 0, inner evaluation budget.
        evaluation_precision: float > 0 or None to reuse final_precision.
        """
        check_precision(final_precision)
        check_positive_integer(max_iterations, "max_iterations")
        check_positive_integer(max_evaluation_iterations, "max_evaluation_iterations")
        if evaluation_precision is not None and (
            not np.isfinite(evaluation_precision) or evaluation_precision <= 0
        ):
            raise ValueError("evaluation_precision must be finite and positive")
        self.model = model
        self.final_precision = final_precision
        self.max_iterations = max_iterations
        self.max_evaluation_iterations = max_evaluation_iterations
        self.evaluation_precision = (
            final_precision if evaluation_precision is None else evaluation_precision
        )
        self.name = __name__.rsplit(".", 1)[-1]

    def run(self):
        """Return the relative bias, policy and native average gain."""
        transitions, rewards = compute_marmote_args(self.model)
        states = MarmoteInterval(0, self.model.state_dim - 1)
        actions = MarmoteInterval(0, self.model.action_dim - 1)
        mdp = AverageMDP("max", states, actions, transitions, rewards)
        solution = mdp.PolicyIterationModified(
            self.final_precision,
            self.max_iterations,
            self.evaluation_precision,
            self.max_evaluation_iterations,
        )
        return relative_result(
            [solution.getValueIndex(s) for s in range(self.model.state_dim)],
            [solution.getActionIndex(s) for s in range(self.model.state_dim)],
            gain=solution.getAvgCost(),
        )
