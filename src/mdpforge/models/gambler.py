import numpy as np
from scipy.sparse import lil_matrix

from mdpforge.core.model import GenericModel


class Model(GenericModel):
    """
    Gambler's Problem finite MDP.

    Source:
        Sutton & Barto, Reinforcement Learning: An Introduction,
        Example 4.3, Gambler's Problem.

    State:
        s = current capital, s in {0, ..., goal}

    Actions:
        action index a corresponds to stake = a + 1.

    Dynamics:
        With probability p_heads:
            s -> s + stake
        With probability 1 - p_heads:
            s -> s - stake

    Reward:
        +1 when reaching the goal state.
        0 otherwise.

    Terminal states:
        0 and goal are absorbing.
    """

    def __init__(self, state_dim: int, action_dim: int):
        # Classical version has states 0, ..., 100.
        # If state_dim=100 is passed by the benchmark CLI, this gives
        # goal=100 and therefore 101 states, matching the textbook.
        self.goal = max(100, state_dim)
        self.state_dim = self.goal + 1

        # Stakes are 1, ..., action_dim.
        # Classical maximum useful stake is goal // 2.
        self.action_dim = max(1, min(action_dim, self.goal // 2))

        self.p_heads = 0.4

        self.name = "{}_{}_gambler".format(self.state_dim, self.action_dim)

    def _build_model(self):
        self.reward_matrix = np.zeros((self.state_dim, self.action_dim))

        self.transition_matrix = [
            lil_matrix((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]

        for s in range(self.state_dim):
            for a in range(self.action_dim):
                stake = a + 1

                if self._is_terminal(s):
                    self.transition_matrix[a][s, s] = 1.0
                    self.reward_matrix[s, a] = 0.0
                    continue

                legal_stake = min(stake, s, self.goal - s)

                win_state = s + legal_stake
                lose_state = s - legal_stake

                self.transition_matrix[a][s, win_state] += self.p_heads
                self.transition_matrix[a][s, lose_state] += 1.0 - self.p_heads

                if win_state == self.goal:
                    self.reward_matrix[s, a] = self.p_heads

        self.transition_matrix = [matrix.tocsr() for matrix in self.transition_matrix]

    def _is_terminal(self, s: int):
        return s == 0 or s == self.goal
