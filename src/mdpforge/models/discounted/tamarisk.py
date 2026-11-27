"""Reduced Tamarisk family inspired by the river-network benchmark, not an
exact IPPC instance. One slot per reach on a directed chain; ternary digit
0=empty, 1=native, 2=invasive. Actions 0=wait, 1..C=eradicate invasive,
C+1..2C=restore native in an empty slot. Eradication succeeds with .8;
restoration succeeds with .7. After intervention, survivors remain with
probability .95 (native) or .90 (invasive). Vacant sites are independently
colonized using competing seed pressures from the original upstream and
 downstream neighbors (downstream dispersal stronger). Natural deaths become
empty for one step. Reward -1 per invasive, -.25 per empty, minus .5/.75
intervention cost. These probabilities and costs are benchmark choices.
state_dim rounds down to 3**C, 1<=C<=7, to limit dense joint transitions.
"""

import numpy as np
from scipy.sparse import csr_matrix

from mdpforge.core.mdp import MDP

METADATA = {
    "category": "asset_management",
    "description": "Reduced stochastic invasive-species management on a directed river "
    "network.",
    "reference": "Managing Invasive Species in a River Network (2012), "
    "https://www.computational-sustainability.org/compsust12/papers/14.pdf; "
    "IPPC 2014 Tamarisk domain.",
    "sizes": {
        "tiny": {"requested_state_dim": 27, "parameters": {}},
        "small": {"requested_state_dim": 81, "parameters": {}},
        "medium": {"requested_state_dim": 243, "parameters": {}},
        "large": {"requested_state_dim": 729, "parameters": {}},
    },
}

TEST_PARAMETERS = {"state_dim": 27}


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
    def __init__(self, state_dim=27):
        self.requested_state_dim = _positive_int(state_dim, "state_dim")
        self.reaches = 1
        while 3 ** (self.reaches + 1) <= self.requested_state_dim:
            self.reaches += 1
        if self.reaches > 7:
            raise ValueError("at most 7 reaches for explicit joint transitions")
        self.state_dim = 3**self.reaches
        self.action_dim = 1 + 2 * self.reaches
        self.name = f"{self.state_dim}_{self.action_dim}_tamarisk"

    def _build_model(self):
        n = self.reaches
        x = _digits(self.state_dim, 3, n)
        native = x == 1
        invasive = x == 2
        native_pressure = np.full(x.shape, 0.05)
        invasive_pressure = np.full(x.shape, 0.05)
        native_pressure[:, 1:] += 0.25 * native[:, :-1]
        native_pressure[:, :-1] += 0.10 * native[:, 1:]
        invasive_pressure[:, 1:] += 0.45 * invasive[:, :-1]
        invasive_pressure[:, :-1] += 0.15 * invasive[:, 1:]
        total = native_pressure + invasive_pressure
        colonized = 1 - np.exp(-total)
        pn = colonized * native_pressure / total
        pi = colonized * invasive_pressure / total
        empty_law = np.stack((1 - pn - pi, pn, pi), axis=2)
        natural = np.where(
            native[:, :, None],
            np.array([0.05, 0.95, 0.0]),
            np.where(invasive[:, :, None], np.array([0.10, 0.0, 0.90]), empty_law),
        )
        reward = -invasive.sum(axis=1) - 0.25 * (x == 0).sum(axis=1)
        self.reward_matrix = np.tile(reward[:, None], (1, self.action_dim))
        self.transition_matrix = []
        for a in range(self.action_dim):
            prob = natural.copy()
            if 1 <= a <= n:
                j = a - 1
                prob[invasive[:, j], j] = (
                    0.2 * np.array([0.10, 0.0, 0.90])
                    + 0.8 * empty_law[invasive[:, j], j]
                )
                self.reward_matrix[:, a] -= 0.5
            elif a > n:
                j = a - n - 1
                mask = x[:, j] == 0
                prob[mask, j] = (
                    0.7 * np.array([0.05, 0.95, 0.0]) + 0.3 * empty_law[mask, j]
                )
                self.reward_matrix[:, a] -= 0.75
            self.transition_matrix.append(_product_matrix(prob, x))
