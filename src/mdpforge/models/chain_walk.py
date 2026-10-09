# Source:
# Lagoudakis, M. G. and Parr, R. (2003).
# Least-Squares Policy Iteration. JMLR 4:1107-1149.
#
# N-state chain walk with two actions. The intended direction succeeds with
# probability 0.9 and is reversed with probability 0.1. Boundary states are
# reflecting. Rewards are obtained on entering either end state.

import numpy as np
from scipy.sparse import dok_array

from mdpforge.core.mdp import MDP

METADATA = {
    "category": "synthetic",
    "description": (
        "Controlled chain walk with uncertain movement and boundary rewards."
    ),
    "reference": (
        "Lagoudakis, M. G. and Parr, R. (2003). Least-Squares Policy Iteration. "
        "JMLR 4:1107-1149."
    ),
}


class Model(MDP):
    LEFT = 0
    RIGHT = 1

    def __init__(self, state_dim: int = 100, action_dim: int = 10):
        del action_dim

        self.state_dim = max(4, int(state_dim))
        self.action_dim = 2
        self.success_probability = 0.9
        self.name = f"{self.state_dim}_{self.action_dim}_chain_walk"

    def _build_model(self) -> None:
        self.reward_matrix = np.zeros(
            (self.state_dim, self.action_dim),
            dtype=np.float64,
        )
        matrices = [
            dok_array((self.state_dim, self.state_dim), dtype=np.float64)
            for _ in range(self.action_dim)
        ]

        for state in range(self.state_dim):
            for action in range(self.action_dim):
                intended_delta = -1 if action == self.LEFT else 1
                opposite_delta = -intended_delta

                intended_state = int(
                    np.clip(state + intended_delta, 0, self.state_dim - 1)
                )
                opposite_state = int(
                    np.clip(state + opposite_delta, 0, self.state_dim - 1)
                )

                matrices[action][state, intended_state] += self.success_probability
                matrices[action][state, opposite_state] += (
                    1.0 - self.success_probability
                )

                reward = 0.0
                if intended_state in (0, self.state_dim - 1):
                    reward += self.success_probability
                if opposite_state in (0, self.state_dim - 1):
                    reward += 1.0 - self.success_probability
                self.reward_matrix[state, action] = reward

        self.transition_matrix = [matrix.tocsr() for matrix in matrices]
