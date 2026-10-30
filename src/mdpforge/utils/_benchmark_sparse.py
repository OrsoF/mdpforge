"""Sparse transition accumulator shared by reduced benchmark recipes."""

import numpy as np
from scipy.sparse import coo_matrix


class TransitionBuilder:
    """Accumulate duplicate transition outcomes and finalize CSR matrices."""

    def __init__(self, states: int, actions: int):
        self.states = states
        self.rows = [[] for _ in range(actions)]
        self.columns = [[] for _ in range(actions)]
        self.probabilities = [[] for _ in range(actions)]
        self.rewards = np.zeros((states, actions), dtype=float)

    def add(self, action: int, state: int, target: int, probability: float):
        if probability:
            self.rows[action].append(state)
            self.columns[action].append(target)
            self.probabilities[action].append(probability)

    def absorbing(self, state: int):
        for action in range(len(self.rows)):
            self.add(action, state, state, 1.0)

    def finish(self):
        return [
            coo_matrix(
                (values, (rows, columns)), shape=(self.states, self.states)
            ).tocsr()
            for rows, columns, values in zip(
                self.rows, self.columns, self.probabilities
            )
        ]
