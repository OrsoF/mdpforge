"""Average-reward relative value iteration through Marmote's public API."""

from marmote.core import MarmoteInterval
from marmote.mdp import AverageMDP

from mdpforge.solvers.average._common import (
    check_positive_integer,
    check_precision,
    relative_result,
)
from mdpforge.utils.solver_backends import compute_marmote_args

METADATA = {
    "display_name": "Average RVI · Marmote",
    "family": "vi",
    "backend": "marmote",
    "description": "Marmote relative value iteration for unichain average-reward MDPs.",
}


class Solver:
    """Run AverageMDP.RelativeValueIteration with its native stopping rule."""

    def __init__(self, model, final_precision=1e-3, *, max_iterations=100_000):
        """Configure native RVI.

        final_precision: finite float > 0, native convergence tolerance.
        max_iterations: int > 0, outer iteration budget.
        """
        check_precision(final_precision)
        check_positive_integer(max_iterations, "max_iterations")
        self.model = model
        self.final_precision = final_precision
        self.max_iterations = max_iterations
        self.name = __name__.rsplit(".", 1)[-1]

    def run(self):
        """Return the relative bias, policy and native average gain."""
        transitions, rewards = compute_marmote_args(self.model)
        states = MarmoteInterval(0, self.model.state_dim - 1)
        actions = MarmoteInterval(0, self.model.action_dim - 1)
        mdp = AverageMDP("max", states, actions, transitions, rewards)
        solution = mdp.RelativeValueIteration(self.final_precision, self.max_iterations)
        return relative_result(
            [solution.getValueIndex(s) for s in range(self.model.state_dim)],
            [solution.getActionIndex(s) for s in range(self.model.state_dim)],
            gain=solution.getAvgCost(),
        )
