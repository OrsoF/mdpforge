# Source:
# Sutton, R. S. and Barto, A. G. (2018).
# Reinforcement Learning: An Introduction, 2nd ed.
# Random-walk prediction examples.
#
# Controlled extension of the classical random walk. The two actions bias the
# walk left or right. The first and last states are absorbing terminal states.

import numpy as np
from scipy.sparse import dok_array

from mdpforge.core.model import GenericModel


class Model(GenericModel):
    LEFT = 0
    RIGHT = 1

    def __init__(self, state_dim: int, action_dim: int):
        del action_dim

        self.state_dim = max(5, int(state_dim))
        self.action_dim = 2
        self.direction_probability = 0.8
        self.name = f"{self.state_dim}_{self.action_dim}_random_walk"

    def _build_model(self) -> None:
        self.reward_matrix = np.zeros(
            (self.state_dim, self.action_dim),
            dtype=np.float64,
        )
        matrices = [
            dok_array((self.state_dim, self.state_dim), dtype=np.float64)
            for _ in range(self.action_dim)
        ]

        left_terminal = 0
        right_terminal = self.state_dim - 1

        for action in range(self.action_dim):
            matrices[action][left_terminal, left_terminal] = 1.0
            matrices[action][right_terminal, right_terminal] = 1.0

        for state in range(1, self.state_dim - 1):
            left_state = state - 1
            right_state = state + 1

            for action in range(self.action_dim):
                p_right = (
                    self.direction_probability
                    if action == self.RIGHT
                    else 1.0 - self.direction_probability
                )
                p_left = 1.0 - p_right

                matrices[action][state, left_state] = p_left
                matrices[action][state, right_state] = p_right

                # Expected immediate reward: -1 at the left terminal,
                # +1 at the right terminal, and 0 otherwise.
                reward = 0.0
                if left_state == left_terminal:
                    reward -= p_left
                if right_state == right_terminal:
                    reward += p_right
                self.reward_matrix[state, action] = reward

        self.transition_matrix = [matrix.tocsr() for matrix in matrices]
