"""IPPC 2014 wildfire on a generated nearly square Moore grid, last cell is
protected target. Two Boolean flags per cell encoded as base-4 digit:
bit 0=burning, bit 1=out of fuel (both can coexist in the original dynamics).
Action 0=wait, 1..C=put out cell, C+1..2C=cut fuel in cell. Cutting the target
has no effect but still costs 5. Ignition is logistic in burning neighbors,
except an isolated target cannot ignite. Burning persists until suppressed;
fuel flag persists or becomes true when currently burning or cut (nontarget).
Reward = -5 per burning nontarget, -100 if target burning or out of fuel,
minus intervention cost (10 suppress/5 cut). state_dim rounds down to 4**C;
1<=C<=6. Only one intervention per step; topology is a generated adaptation.
"""

from math import ceil, sqrt

import numpy as np
from scipy.sparse import csr_matrix

from mdpforge.core.mdp import MDP

METADATA = {
    "category": "asset_management",
    "description": "Stochastic spatial fire propagation with suppression and fuel "
    "removal.",
    "reference": "Zhenyu Yu, IPPC 2014 Wildfire MDP; Karafyllidis & Thanailakis "
    "(1997), Ecological Modelling 99, 87–97. "
    "https://github.com/pyrddlgym-project/rddlrepository/tree/main/rddlrepository/archive/competitions/IPPC2014/Wildfire/MDP",
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
    def __init__(self, state_dim=16):
        self.requested_state_dim = _positive_int(state_dim, "state_dim")
        self.cells = max(1, (self.requested_state_dim.bit_length() - 1) // 2)
        if self.cells > 6:
            raise ValueError("at most 6 cells for explicit joint transitions")
        self.state_dim = 4**self.cells
        self.action_dim = 1 + 2 * self.cells
        self.name = f"{self.state_dim}_{self.action_dim}_wildfire"

    def _build_model(self):
        n = self.cells
        digits = _digits(self.state_dim, 4, n)
        burning = (digits & 1) != 0
        fuel_gone = (digits & 2) != 0
        neighbors = _neighbors(n, ceil(sqrt(n)))
        count = np.column_stack([burning[:, list(nb)].sum(axis=1) for nb in neighbors])
        ignition = 1 / (1 + np.exp(4.5 - count))
        ignition[count[:, -1] == 0, -1] = 0
        base_p = np.where(burning, 1.0, np.where(fuel_gone, 0.0, ignition))
        cost = -5 * burning[:, :-1].sum(axis=1) - 100 * (
            burning[:, -1] | fuel_gone[:, -1]
        )
        self.reward_matrix = np.tile(cost[:, None], (1, self.action_dim)).astype(float)
        self.transition_matrix = []
        for a in range(self.action_dim):
            p = base_p.copy()
            gone = fuel_gone | burning
            if 1 <= a <= n:
                p[:, a - 1] = 0
                self.reward_matrix[:, a] -= 10
            elif a > n:
                j = a - n - 1
                if j != n - 1:
                    gone[:, j] = True
                self.reward_matrix[:, a] -= 5
            prob = np.stack(
                ((1 - p) * ~gone, p * ~gone, (1 - p) * gone, p * gone), axis=2
            )
            self.transition_matrix.append(_product_matrix(prob, digits))
