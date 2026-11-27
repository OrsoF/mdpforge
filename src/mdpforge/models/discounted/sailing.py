"""Vanderbei-style sailing: square grid, eight wind directions (direction FROM),
previous port/starboard tack, and one absorbing goal at the last grid cell.
Eight compass headings; wind rotates +/-45 degrees with probability 0.2 each,
otherwise remains unchanged. Leg times 4/3/2/1 for relative angle 45/90/135/180
 degrees, multiplied by sqrt(2) for diagonal legs, plus 3 when switching tack.
Direct upwind and off-grid actions wait for a wind change at cost 8. Downwind
retains previous tack. These waiting and downwind conventions are adaptations.
Reward is minus elapsed time. state_dim rounds down to 16*n*n+1 (min n=2);
goal-cell encodings also absorb into the common terminal with zero reward.
"""

from math import isqrt

import numpy as np
from scipy.sparse import csr_matrix

from mdpforge.core.mdp import MDP

METADATA = {
    "category": "navigation",
    "description": "Grid sailing with stochastic wind and tack switching costs.",
    "reference": "Vanderbei, Sailing Strategies, Princeton University. "
    "https://vanderbei.princeton.edu/sail/sail.html",
    "sizes": {
        "tiny": {"requested_state_dim": 401, "parameters": {}},
        "small": {"requested_state_dim": 3601, "parameters": {}},
        "medium": {"requested_state_dim": 32401, "parameters": {}},
        "large": {"requested_state_dim": 160001, "parameters": {}},
    },
}

TEST_PARAMETERS = {"state_dim": 145}


def _positive_int(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


class Model(MDP):
    def __init__(self, state_dim=401):
        self.requested_state_dim = _positive_int(state_dim, "state_dim")
        self.side = max(2, isqrt((self.requested_state_dim - 1) // 16))
        self.state_dim = 16 * self.side**2 + 1
        self.action_dim = 8
        self.terminal = self.state_dim - 1
        self.name = f"{self.state_dim}_8_sailing"

    def _build_model(self):
        n = self.side
        s = np.arange(self.terminal, dtype=np.int64)
        cell, wind, tack = s // 16, (s // 2) % 8, s % 2
        x, y = cell // n, cell % n
        goal = cell == n * n - 1
        self.reward_matrix = np.zeros((self.state_dim, 8))
        self.transition_matrix = []
        directions = (
            (-1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
            (1, 0),
            (1, -1),
            (0, -1),
            (-1, -1),
        )
        for a, (dx, dy) in enumerate(directions):
            rel = (a - wind) % 8
            angle = np.minimum(rel, 8 - rel)
            valid = (
                (angle != 0)
                & (x + dx >= 0)
                & (x + dx < n)
                & (y + dy >= 0)
                & (y + dy < n)
            )
            nt = np.where(valid & (rel != 4), (rel > 4).astype(int), tack)
            nc = np.where(valid, (x + dx) * n + y + dy, cell)
            cost = np.where(
                valid, (5 - angle) * np.hypot(dx, dy) + 3 * (nt != tack), 8.0
            )
            self.reward_matrix[:-1, a] = np.where(goal, 0, -cost)
            done = goal | (nc == n * n - 1)
            rr = np.concatenate((s, s, s, [self.terminal]))
            cc = np.concatenate(
                [
                    np.where(done, self.terminal, 16 * nc + 2 * ((wind + d) % 8) + nt)
                    for d in (-1, 0, 1)
                ]
                + [[self.terminal]]
            )
            pp = np.concatenate(
                (
                    np.full(len(s), 0.2),
                    np.full(len(s), 0.6),
                    np.full(len(s), 0.2),
                    [1.0],
                )
            )
            self.transition_matrix.append(
                csr_matrix((pp, (rr, cc)), shape=(self.state_dim,) * 2)
            )
