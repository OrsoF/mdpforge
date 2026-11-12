"""Total-reward VI · Marmote · GS through Marmote's public API."""

import numpy as np
from marmote.core import MarmoteInterval
from marmote.mdp import TotalRewardMDP

from mdpforge.core.solve_result import SolveResult
from mdpforge.solvers.total._common import check_options
from mdpforge.utils.solver_backends import compute_marmote_args

METADATA = {
    "display_name": "Total VI · Marmote · GS",
    "family": "vi",
    "backend": "marmote",
    "description": "Total-reward ValueIterationGS using Marmote.",
}


class Solver:
    """Call Marmote's total-reward ValueIterationGS method."""

    def __init__(
        self,
        model,
        final_precision=1e-3,
        *,
        max_iterations=100_000,
    ):
        check_options(model, final_precision, max_iterations)
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
        solution = mdp.ValueIterationGS(self.final_precision, self.max_iterations)
        return SolveResult(
            np.array([solution.getValueIndex(s) for s in range(self.model.state_dim)]),
            np.array([solution.getActionIndex(s) for s in range(self.model.state_dim)]),
            validation_method="residual",
        )
