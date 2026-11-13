"""Modified policy iteration for total-reward MDPs."""

import numpy as np

from mdpforge.core.operators import bellman_operator, compute_transition_reward_policy
from mdpforge.core.solve_result import SolveResult
from mdpforge.solvers.total._common import check_options, proper_initial_policy

METADATA = {
    "display_name": "Total MPI · mdpforge",
    "family": "mpi",
    "backend": "mdpforge",
    "description": "Total-reward modified policy iteration with truncated evaluation.",
}


class Solver:
    """Alternate truncated policy evaluation and greedy improvement."""

    def __init__(
        self,
        model,
        final_precision=1e-3,
        *,
        max_iterations=100_000,
        max_evaluation_iterations=10,
    ):
        self.terminal = check_options(model, final_precision, max_iterations)
        if type(max_evaluation_iterations) is not int or max_evaluation_iterations < 1:
            raise ValueError("max_evaluation_iterations must be positive")
        self.model = model
        self.final_precision = final_precision
        self.max_iterations = max_iterations
        self.max_evaluation_iterations = max_evaluation_iterations
        self.name = __name__.rsplit(".", 1)[-1]

    def run(self):
        """Return values and policy when the full residual is small."""
        policy = proper_initial_policy(
            self.model, self.terminal, require_all_states=False
        )
        value = np.zeros(self.model.state_dim)
        for _ in range(self.max_iterations):
            transition, reward = compute_transition_reward_policy(self.model, policy)
            for _ in range(self.max_evaluation_iterations):
                value = reward + transition.dot(value)
            q_value = bellman_operator(self.model, value, 1.0)
            if not np.isfinite(q_value).all():
                raise RuntimeError("MPI stopped: non-finite values")
            improved = q_value.argmax(axis=1)
            if np.max(np.abs(q_value.max(axis=1) - value)) <= self.final_precision:
                return SolveResult(value, improved, validation_method="residual")
            policy = improved
        raise RuntimeError("MPI stopped: max_iterations reached")
