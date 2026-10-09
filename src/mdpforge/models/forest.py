# Source: classical forest-management replacement benchmark.

import numpy as np
from scipy.sparse import lil_matrix

from mdpforge.core.mdp import MDP


class Model(MDP):
    def __init__(self, state_dim: int = 100, action_dim: int = 10):
        self.state_dim = max(3, int(state_dim))
        self.action_dim = 2

        self.fire_probability = 0.1
        self.wait_reward = 1.0
        self.mature_wait_reward = 4.0
        self.cut_reward = 2.0

        self.name = "{}_{}_forest".format(self.state_dim, self.action_dim)

    def _build_model(self):
        self.transition_matrix = [
            lil_matrix((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]
        self.reward_matrix = np.zeros((self.state_dim, self.action_dim))

        wait = 0
        cut = 1
        mature_state = self.state_dim - 1

        for state in range(self.state_dim):
            next_age = min(state + 1, mature_state)

            self.transition_matrix[wait][state, 0] += self.fire_probability
            self.transition_matrix[wait][state, next_age] += 1.0 - self.fire_probability
            self.reward_matrix[state, wait] = (
                self.mature_wait_reward if state == mature_state else self.wait_reward
            )

            self.transition_matrix[cut][state, 0] = 1.0
            self.reward_matrix[state, cut] = (
                self.cut_reward * (state + 1) / self.state_dim
            )

        self.transition_matrix = [
            transition.tocsr() for transition in self.transition_matrix
        ]
