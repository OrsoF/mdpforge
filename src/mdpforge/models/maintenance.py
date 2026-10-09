# Source: stochastic machine-maintenance and deterioration benchmark.

import numpy as np
from scipy.sparse import lil_matrix

from mdpforge.core.mdp import MDP


class Model(MDP):
    def __init__(self, state_dim: int = 100, action_dim: int = 10):
        self.state_dim = max(5, int(state_dim))
        self.action_dim = 4

        self.name = "{}_{}_maintenance".format(self.state_dim, self.action_dim)

    def _build_model(self):
        self.transition_matrix = [
            lil_matrix((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]
        self.reward_matrix = np.zeros((self.state_dim, self.action_dim))

        do_nothing = 0
        inspect = 1
        repair = 2
        replace = 3
        failed = self.state_dim - 1

        for state in range(self.state_dim):
            condition = state / failed
            operating_reward = 2.0 * (1.0 - condition)
            failure_penalty = 4.0 if state == failed else 0.0

            self.reward_matrix[state, do_nothing] = operating_reward - failure_penalty
            self._add_deterioration(do_nothing, state, drift=1)

            self.reward_matrix[state, inspect] = (
                operating_reward - 0.25 - failure_penalty
            )
            self._add_deterioration(inspect, state, drift=0)

            repaired_state = max(0, state - max(1, self.state_dim // 5))
            self.reward_matrix[state, repair] = operating_reward - 1.0
            self.transition_matrix[repair][state, repaired_state] += 0.8
            self.transition_matrix[repair][state, min(failed, repaired_state + 1)] += (
                0.2
            )

            self.reward_matrix[state, replace] = -2.0
            self.transition_matrix[replace][state, 0] = 1.0

        self.transition_matrix = [
            transition.tocsr() for transition in self.transition_matrix
        ]

    def _add_deterioration(self, action: int, state: int, drift: int) -> None:
        failed = self.state_dim - 1
        same = state
        worse = min(failed, state + 1 + drift)
        much_worse = min(failed, state + 2 + drift)

        self.transition_matrix[action][state, same] += 0.55
        self.transition_matrix[action][state, worse] += 0.35
        self.transition_matrix[action][state, much_worse] += 0.10
