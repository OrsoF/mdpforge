# Source:
# Sutton, R. S. and Barto, A. G. (2018).
# Reinforcement Learning: An Introduction, 2nd ed., Example 10.2.
#
# Access-control queueing task. A state is (number of free servers,
# priority class of the arriving customer). The agent accepts or rejects.
# This is naturally an average-reward continuing MDP.

import math

import numpy as np
from scipy.sparse import dok_array

from mdpforge.core.mdp import MDP


class Model(MDP):
    REJECT = 0
    ACCEPT = 1

    def __init__(self, state_dim: int = 400, action_dim: int = 10):
        del action_dim

        state_dim = int(np.sqrt(state_dim))
        requested = max(20, int(state_dim))
        self.num_priorities = 4
        self.num_servers = max(1, requested // self.num_priorities - 1)

        self.priority_rewards = np.array([1.0, 2.0, 4.0, 8.0], dtype=np.float64)
        self.free_probability = 0.06
        self.action_dim = 2
        self.state_dim = (self.num_servers + 1) * self.num_priorities
        self.name = f"{self.state_dim}_{self.action_dim}_access_control"

    def _encode(self, free_servers: int, priority: int) -> int:
        return free_servers * self.num_priorities + priority

    def _build_model(self) -> None:
        self.reward_matrix = np.zeros(
            (self.state_dim, self.action_dim),
            dtype=np.float64,
        )
        matrices = [
            dok_array((self.state_dim, self.state_dim), dtype=np.float64)
            for _ in range(self.action_dim)
        ]

        priority_probability = 1.0 / self.num_priorities

        for free_servers in range(self.num_servers + 1):
            busy_servers = self.num_servers - free_servers

            for priority in range(self.num_priorities):
                state = self._encode(free_servers, priority)

                for action in range(self.action_dim):
                    accepted = action == self.ACCEPT and free_servers > 0
                    free_after_decision = free_servers - int(accepted)
                    busy_after_decision = busy_servers + int(accepted)

                    self.reward_matrix[state, action] = (
                        self.priority_rewards[priority] if accepted else 0.0
                    )

                    # During the step, each busy server independently becomes
                    # free with probability free_probability.
                    for released in range(busy_after_decision + 1):
                        release_probability = (
                            math.comb(busy_after_decision, released)
                            * self.free_probability**released
                            * (1.0 - self.free_probability)
                            ** (busy_after_decision - released)
                        )
                        next_free = min(
                            self.num_servers,
                            free_after_decision + released,
                        )

                        for next_priority in range(self.num_priorities):
                            next_state = self._encode(next_free, next_priority)
                            matrices[action][state, next_state] += (
                                release_probability * priority_probability
                            )

        self.transition_matrix = [matrix.tocsr() for matrix in matrices]
