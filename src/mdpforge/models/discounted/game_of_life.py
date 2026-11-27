"""IPPC 2011 controlled stochastic Game of Life; generated nearly square grid
with a possibly incomplete final row and Moore neighborhood (no wrapping).
State is a binary mask of live cells; action 0=no intervention, j+1=force
cell j's intended next value to alive. Conway survival/birth is followed by
independent bit noise of probability noise in [0,0.5]. Reward = current live
cell count minus 1 for intervention. state_dim rounds down to 2**cells;
1<=cells<=10 (explicit dense transition memory guard).
"""

from math import ceil, sqrt

import numpy as np
from scipy.sparse import csr_matrix

from mdpforge.core.mdp import MDP

METADATA = {
    "category": "games_and_puzzles",
    "description": "Noisy Conway automaton with single-cell intervention.",
    "reference": "Scott Sanner, IPPC 2011 Game of Life MDP. "
    "https://github.com/pyrddlgym-project/rddlrepository/tree/main/rddlrepository/archive/competitions/IPPC2011/GameOfLife/MDP",
    "sizes": {
        "tiny": {"requested_state_dim": 16, "parameters": {}},
        "small": {"requested_state_dim": 64, "parameters": {}},
        "medium": {"requested_state_dim": 256, "parameters": {}},
        "large": {"requested_state_dim": 1024, "parameters": {}},
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


def _digits(n, base, count):
    return (np.arange(n, dtype=np.int64)[:, None] // (base ** np.arange(count))) % base


def _neighbors(count, width):
    return [
        tuple(
            j
            for j in range(count)
            if j != i
            and abs(j // width - i // width) <= 1
            and abs(j % width - i % width) <= 1
        )
        for i in range(count)
    ]


def _product_matrix(probabilities, digits):
    """Independent local outcomes; enumerate their exact joint law one row at a time."""
    n, count, _ = probabilities.shape
    indptr = np.zeros(n + 1, dtype=np.int64)
    indices = []
    values = []
    for s in range(n):
        p = np.prod(probabilities[s, np.arange(count), digits], axis=1)
        nonzero = np.flatnonzero(p > 0)
        indices.append(nonzero.astype(np.int32))
        values.append(p[nonzero])
        indptr[s + 1] = indptr[s] + len(nonzero)
    return csr_matrix(
        (np.concatenate(values), np.concatenate(indices), indptr), shape=(n, n)
    )


class Model(MDP):
    def __init__(self, state_dim=16, noise=0.1):
        self.requested_state_dim = _positive_int(state_dim, "state_dim")
        self.cells = max(1, self.requested_state_dim.bit_length() - 1)
        if self.cells > 10:
            raise ValueError("at most 10 cells for explicit joint transitions")
        if not np.isfinite(noise) or not 0 <= noise <= 0.5:
            raise ValueError("noise must be in [0,0.5]")
        self.noise = float(noise)
        self.state_dim = 2**self.cells
        self.action_dim = self.cells + 1
        self.name = f"{self.state_dim}_{self.action_dim}_gameoflife_{self.noise}"

    def _build_model(self):
        x = _digits(self.state_dim, 2, self.cells)
        neighbors = _neighbors(self.cells, ceil(sqrt(self.cells)))
        count = np.column_stack([x[:, list(nb)].sum(axis=1) for nb in neighbors])
        intended = (count == 3) | ((x == 1) & (count == 2))
        self.reward_matrix = np.tile(
            x.sum(axis=1)[:, None], (1, self.action_dim)
        ).astype(float)
        self.reward_matrix[:, 1:] -= 1
        self.transition_matrix = []
        for a in range(self.action_dim):
            p = np.where(intended, 1 - self.noise, self.noise)
            if a:
                p[:, a - 1] = 1 - self.noise
            self.transition_matrix.append(
                _product_matrix(np.stack((1 - p, p), axis=2), x)
            )
