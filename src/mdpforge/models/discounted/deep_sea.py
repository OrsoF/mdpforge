"""Expected-reward version of bsuite DeepSea (https://github.com/google-deepmind/bsuite).
State (row, column), 0 <= column <= row < n, encoded row*(row+1)//2+column;
last state absorbs with zero reward. Two actions map to left/right by seeded
random labels. Right costs 0.01/n and pays 1 in the final rightmost cell;
windy right moves fail by staying in the column with probability 1/n.
Zero-mean reward noise is integrated out. state_dim rounds down to a triangular
number plus one (minimum n=2). Discounting changes the original episodic objective.
"""

from math import isqrt

import numpy as np
from scipy.sparse import csr_matrix

from mdpforge.core.mdp import MDP

METADATA = {
    "category": "abstract",
    "description": "Deep exploration with state-dependent action labels and sparse "
    "terminal rewards.",
    "reference": "Osband et al. (2020), Behaviour Suite for Reinforcement Learning, "
    "ICLR. https://arxiv.org/abs/1908.03568",
    "sizes": {
        "tiny": {"requested_state_dim": 821, "parameters": {}},
        "small": {"requested_state_dim": 5051, "parameters": {}},
        "medium": {"requested_state_dim": 31376, "parameters": {}},
        "large": {"requested_state_dim": 125251, "parameters": {}},
    },
}

TEST_PARAMETERS = {"state_dim": 16}


def _positive_int(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


class Model(MDP):
    def __init__(self, state_dim=821, seed=0, stochastic=False):
        self.requested_state_dim = _positive_int(state_dim, "state_dim")
        if (
            isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, (int, np.integer))
            or seed < 0
        ):
            raise ValueError("seed must be a nonnegative integer")
        if not isinstance(stochastic, (bool, np.bool_)):
            raise ValueError("stochastic must be boolean")
        self.seed = int(seed)
        self.stochastic = bool(stochastic)
        self.side = max(2, (isqrt(8 * (self.requested_state_dim - 1) + 1) - 1) // 2)
        self.state_dim = self.side * (self.side + 1) // 2 + 1
        self.action_dim = 2
        self.terminal = self.state_dim - 1
        self.name = f"{self.state_dim}_2_deepsea_{self.seed}_{int(self.stochastic)}"

    def _build_model(self):
        n = self.side
        s = np.arange(self.terminal, dtype=np.int64)
        row = np.repeat(np.arange(n), np.arange(1, n + 1))
        col = s - row * (row + 1) // 2
        offset = (row + 1) * (row + 2) // 2
        final = row == n - 1
        labels = np.random.default_rng(self.seed).integers(0, 2, size=self.terminal)
        self.reward_matrix = np.zeros((self.state_dim, 2))
        self.transition_matrix = []
        for a in range(2):
            right = labels == a
            next_col = np.where(right, col + 1, np.maximum(col - 1, 0))
            dest = np.where(final, self.terminal, offset + next_col)
            failure = right & ~final & self.stochastic
            prob = np.where(failure, 1 - 1 / n, 1.0)
            rr = np.concatenate((s, s[failure], [self.terminal]))
            cc = np.concatenate((dest, (offset + col)[failure], [self.terminal]))
            pp = np.concatenate((prob, np.full(failure.sum(), 1 / n), [1.0]))
            self.transition_matrix.append(
                csr_matrix((pp, (rr, cc)), shape=(self.state_dim,) * 2)
            )
            self.reward_matrix[:-1, a] = right * (-0.01 / n + (final & (col == n - 1)))
