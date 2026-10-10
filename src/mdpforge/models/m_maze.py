# Source: McCallum (1996), U-Tree / UDM aliased maze variants.
# M-maze implemented as a discrete POMDP-style grid with local observations.

import numpy as np
from scipy.sparse import lil_matrix

from mdpforge.core.mdp import MDP

METADATA = {
    "tags": ["navigation", "maze"],
    "sizes": {
        "small": {
            "state_dim": 100,
            "parameters": {"state_dim": 100},
            "source": "measured",
        },
        "medium": {
            "state_dim": 1023,
            "parameters": {"state_dim": 1000},
            "source": "measured",
        },
        "large": {
            "state_dim": 6723,
            "parameters": {"state_dim": 6674},
            "source": "inferred",
        },
    },
    "category": "navigation",
    "description": "M-shaped maze with aliased local observations.",
    "reference": "McCallum (1996), U-Tree / UDM aliased maze variants.",
}


class Model(MDP):
    """M-shaped aliased maze.

    Actions: 0=N, 1=S, 2=E, 3=W.  Concrete state is grid position; observations
    are adjacent-wall bits plus a weak corridor parity bit to preserve aliasing.
    """

    def __init__(self, state_dim: int = 81, action_dim: int = 10):
        self.height = max(7, int(np.sqrt(max(49, state_dim))))
        self.width = max(9, int(np.ceil(max(81, state_dim) / self.height)))
        self.action_dim = 4
        self.state_dim = self.height * self.width
        self.observation_dim = 32
        self.name = f"{self.state_dim}_{self.action_dim}_m_maze"

    def _build_model(self):
        self.blocked = self._make_m_walls()
        self.goal_cell = self._cell(1, self.width - 2)
        self.reward_matrix = -np.ones((self.state_dim, self.action_dim), dtype=float)
        self.transition_matrix = [
            lil_matrix((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]
        self.observation_map = np.zeros(self.state_dim, dtype=int)

        for s in range(self.state_dim):
            self.observation_map[s] = self._observe(s)
            if s == self.goal_cell:
                for a in range(self.action_dim):
                    self.transition_matrix[a][s, s] = 1.0
                    self.reward_matrix[s, a] = 0.0
                continue
            for a in range(self.action_dim):
                intended = self._move(s, a)
                self.transition_matrix[a][s, intended] += 0.9
                self.transition_matrix[a][s, s] += 0.1
                self.reward_matrix[s, a] = 25.0 if intended == self.goal_cell else -1.0

        self.observation_matrix = np.zeros(
            (self.state_dim, self.observation_dim), dtype=float
        )
        self.observation_matrix[np.arange(self.state_dim), self.observation_map] = 1.0
        self.transition_matrix = [m.tocsr() for m in self.transition_matrix]

    def _cell(self, i, j):
        return i * self.width + j

    def _coord(self, s):
        return s // self.width, s % self.width

    def _make_m_walls(self):
        blocked = set()
        for i in range(self.height):
            blocked.add(self._cell(i, 0))
            blocked.add(self._cell(i, self.width - 1))
        for j in range(self.width):
            blocked.add(self._cell(0, j))
            blocked.add(self._cell(self.height - 1, j))
        left, mid, right = self.width // 4, self.width // 2, 3 * self.width // 4
        for i in range(1, self.height - 1):
            if i != self.height - 2:
                blocked.add(self._cell(i, left))
            if i != 1:
                blocked.add(self._cell(i, mid))
            if i != self.height - 2:
                blocked.add(self._cell(i, right))
        # diagonal strokes of the M, with periodic gaps to avoid dead partitions
        for k in range(1, min(self.height - 1, self.width // 4)):
            c1 = left + k
            c2 = right - k
            if k % 3 != 0:
                if c1 < mid:
                    blocked.add(self._cell(k, c1))
                if c2 > mid:
                    blocked.add(self._cell(k, c2))
        blocked.discard(self.goal_cell if hasattr(self, "goal_cell") else -1)
        return blocked

    def _neighbor(self, s, a):
        i, j = self._coord(s)
        di, dj = [(-1, 0), (1, 0), (0, 1), (0, -1)][a]
        ni, nj = i + di, j + dj
        if not (0 <= ni < self.height and 0 <= nj < self.width):
            return s
        return self._cell(ni, nj)

    def _move(self, s, a):
        ns = self._neighbor(s, a)
        return s if ns in self.blocked else ns

    def _observe(self, s):
        i, j = self._coord(s)
        obs = 0
        for bit, a in enumerate([0, 1, 2, 3]):
            ns = self._neighbor(s, a)
            if ns == s or ns in self.blocked:
                obs |= 1 << bit
        obs |= (j % 2) << 4
        return obs
