"""Marmote value iteration for absorbing total-reward MDPs."""

import numpy as np
from marmote.core import MarmoteInterval
from marmote.mdp import TotalRewardMDP

from mdpforge.core.solve_result import SolveResult
from mdpforge.utils.solver_backends import compute_marmote_args

METADATA = {
    "display_name": "Total VI · Marmote",
    "family": "vi",
    "backend": "marmote",
    "description": "Marmote total-reward value iteration.",
}


class Solver:
    """Call Marmote's public total-reward value iteration API."""

    def __init__(self, model, final_precision=1e-3, *, max_iterations=100_000):
        if not np.isfinite(final_precision) or final_precision <= 0:
            raise ValueError("final_precision must be finite and positive")
        if type(max_iterations) is not int or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        self.model = model
        self.final_precision = final_precision
        self.max_iterations = max_iterations
        self.name = __name__.rsplit(".", 1)[-1]

    def run(self):
        """Return Marmote's value vector and policy."""
        transitions, rewards = compute_marmote_args(self.model)
        states = MarmoteInterval(0, int(self.model.state_dim - 1))
        actions = MarmoteInterval(0, int(self.model.action_dim - 1))
        mdp = TotalRewardMDP("max", states, actions, transitions, rewards)
        solution = mdp.ValueIteration(self.final_precision, self.max_iterations)
        return SolveResult(
            np.array([solution.getValueIndex(s) for s in range(self.model.state_dim)]),
            np.array([solution.getActionIndex(s) for s in range(self.model.state_dim)]),
            validation_method="residual",
        )
