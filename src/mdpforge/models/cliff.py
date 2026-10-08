# Source: Sutton, R. S. and Barto, A. G. (2018). Reinforcement Learning: An Introduction, 2nd ed., Example 6.6.
import numpy as np
from scipy.sparse import dok_array

from mdpforge.core.model import GenericModel


class Model(GenericModel):
    def __init__(self, state_dim: int, action_dim: int):
        self.action_dim = 4
        self.v_dim = int(np.sqrt(state_dim / 3))
        self.h_dim = 3 * self.v_dim
        self.state_dim = self.h_dim * self.v_dim

        self.name = "{}_{}_cliff".format(
            self.state_dim,
            self.action_dim,
        )

    def is_in_hole(self, state_index) -> bool:
        v = state_index // self.h_dim
        h = state_index % self.h_dim
        return v == 0 and h > 0 and h < self.h_dim - 1

    def get_next_state(self, state_index: int, action: int) -> int:
        v = state_index // self.h_dim
        h = state_index % self.h_dim

        if action == 0:
            next_v = min(v + 1, self.v_dim - 1)
            next_h = h

        elif action == 1:
            next_v = max(v - 1, 0)
            next_h = h

        elif action == 2:
            next_v = v
            next_h = min(h + 1, self.h_dim - 1)

        else:
            next_v = v
            next_h = max(h - 1, 0)

        return next_v * self.h_dim + next_h

    def _build_model(self):
        self.reward_matrix = np.zeros(
            (self.state_dim, self.action_dim), dtype=np.float32
        )
        self.transition_matrix = [
            dok_array((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]

        start_state = 0
        goal_state = self.h_dim - 1

        for state_index in range(self.state_dim):
            for action in range(self.action_dim):
                # Goal is absorbing with zero future reward.
                if state_index == goal_state:
                    self.reward_matrix[state_index, action] = 0.0
                    self.transition_matrix[action][state_index, goal_state] = 1.0
                    continue

                next_state = self.get_next_state(state_index, action)

                # Stepping into the cliff gives immediate penalty and resets to start.
                if self.is_in_hole(next_state):
                    self.reward_matrix[state_index, action] = -100.0
                    self.transition_matrix[action][state_index, start_state] = 1.0

                # Reaching the goal ends the episode / enters absorbing goal.
                elif next_state == goal_state:
                    self.reward_matrix[state_index, action] = -1.0
                    self.transition_matrix[action][state_index, goal_state] = 1.0

                # Normal movement cost.
                else:
                    self.reward_matrix[state_index, action] = -1.0
                    self.transition_matrix[action][state_index, next_state] = 1.0

        self.transition_matrix = [matrix.tocsr() for matrix in self.transition_matrix]
