# Source: Hyon, E. (2020). Apprehender le hasard. Tangente Magazine.

import numpy as np
from scipy.sparse import csr_matrix

from core.model import GenericModel


class Model(GenericModel):
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = max(2, state_dim)
        self.action_dim = 3

        self.name = "{}_{}_schoolboy".format(self.state_dim, self.action_dim)

    def _build_model(self):
        self.transition_matrix = []
        self.reward_matrix = np.full((self.state_dim, self.action_dim), -15.0)

        speeds = [
            [(30, 0.4), (45, 0.6)],  # walk
            [(20, 0.7), (35, 0.3)],  # bike
            [(15, 0.5), (45, 0.5)],  # bus
        ]

        for aa in range(self.action_dim):
            rows, cols, data = [], [], []

            for ss in range(self.state_dim):
                if ss == self.state_dim - 1:
                    rows.append(ss)
                    cols.append(ss)
                    data.append(1.0)
                    continue

                for time, probability in speeds[aa]:
                    step = max(1, round((self.state_dim - 1) * 15 / time))
                    next_state = min(self.state_dim - 1, ss + step)

                    rows.append(ss)
                    cols.append(next_state)
                    data.append(probability)

            self.transition_matrix.append(
                csr_matrix((data, (rows, cols)), shape=(self.state_dim, self.state_dim))
            )

        self.reward_matrix[-1, :] = 0.0
