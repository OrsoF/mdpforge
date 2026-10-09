# Source: McCallum (1996), Reinforcement Learning with Selective Perception and Hidden State.
# Local-perception maze: concrete grid state, aliased adjacent-wall observations.

import numpy as np
from scipy.sparse import lil_matrix

from mdpforge.core.mdp import MDP


class Model(MDP):
    """Aliased gridworld with local wall-bit observations.

    Actions: 0=N, 1=S, 2=E, 3=W.  State is the true cell.  Observations are not
    part of the Markov state; use `observation_map` or `observation_matrix`.
    Observation bit order: N,S,E,W wall-present.
    """

    def __init__(self, state_dim: int = 49, action_dim: int = 10):
        self.height = max(7, int(np.sqrt(max(49, state_dim))))
        self.width = max(7, int(np.ceil(max(49, state_dim) / self.height)))
        self.action_dim = 4
        self.state_dim = self.height * self.width
        self.observation_dim = 16
        self.name = f"{self.state_dim}_{self.action_dim}_local_perception_grid_world"

    def _build_model(self):
        self.blocked = self._make_walls()
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
                ns = self._move(s, a)
                self.transition_matrix[a][s, ns] = 1.0
                self.reward_matrix[s, a] = 20.0 if ns == self.goal_cell else -1.0

        self.observation_matrix = np.zeros(
            (self.state_dim, self.observation_dim), dtype=float
        )
        self.observation_matrix[np.arange(self.state_dim), self.observation_map] = 1.0
        self.transition_matrix = [m.tocsr() for m in self.transition_matrix]

    def _cell(self, i, j):
        return i * self.width + j

    def _coord(self, s):
        return s // self.width, s % self.width

    def _make_walls(self):
        blocked = set()
        for i in range(self.height):
            blocked.add(self._cell(i, 0))
            blocked.add(self._cell(i, self.width - 1))
        for j in range(self.width):
            blocked.add(self._cell(0, j))
            blocked.add(self._cell(self.height - 1, j))
        for j in range(2, self.width - 2):
            if j % 3 != 0:
                blocked.add(self._cell(2, j))
        for j in range(2, self.width - 2):
            if j % 3 != 1:
                blocked.add(self._cell(self.height - 3, j))
        for i in range(2, self.height - 2):
            if i % 3 != 2:
                blocked.add(self._cell(i, self.width // 2))
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
        obs = 0
        for bit, a in enumerate([0, 1, 2, 3]):
            ns = self._neighbor(s, a)
            if ns == s or ns in self.blocked:
                obs |= 1 << bit
        return obs
