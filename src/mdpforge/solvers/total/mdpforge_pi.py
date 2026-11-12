"""Policy iteration for total-reward MDPs."""

import numpy as np
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve

from mdpforge.core.operators import bellman_operator, compute_transition_reward_policy
from mdpforge.core.solve_result import SolveResult
from mdpforge.solvers.total._common import check_options, proper_initial_policy

METADATA = {
    "display_name": "Total PI · mdpforge",
    "family": "pi",
    "backend": "mdpforge",
    "description": "Total-reward policy iteration with sparse policy evaluation.",
}


class Solver:
    """Evaluate and improve a proper policy until stable."""

    def __init__(self, model, *, max_iterations=100_000):
        self.terminal = check_options(model, 1.0, max_iterations)
        self.model = model
        self.max_iterations = max_iterations
        self.name = __name__.rsplit(".", 1)[-1]

    def run(self):
        """Return the value and a stable policy."""
        transient = np.array(
            [s for s in range(self.model.state_dim) if s not in self.terminal]
        )
        policy = proper_initial_policy(self.model, self.terminal)
        identity = eye(len(transient), format="csr")
        for _ in range(self.max_iterations):
            transition, reward = compute_transition_reward_policy(self.model, policy)
            value = np.zeros(self.model.state_dim)
            value[transient] = spsolve(
                identity - transition[transient, :][:, transient], reward[transient]
            )
            if not np.isfinite(value).all():
                raise RuntimeError("PI stopped: policy evaluation is singular")
            q_value = bellman_operator(self.model, value, 1.0)
            improved = q_value.argmax(axis=1)
            states = np.arange(self.model.state_dim)
            keep = np.isclose(
                q_value[states, policy],
                q_value[states, improved],
                rtol=0.0,
                atol=1e-12,
            )
            improved[keep] = policy[keep]
            if np.array_equal(improved, policy):
                return SolveResult(value, policy, validation_method="residual")
            policy = improved
        raise RuntimeError("PI stopped: max_iterations reached")
