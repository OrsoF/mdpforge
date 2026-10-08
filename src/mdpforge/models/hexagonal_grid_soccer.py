# Source: Continuous U-Tree, Uther & Veloso (1998), enlarged hexagonal-grid soccer;
# inspired by Littman (1994) Markov games.  Single-agent tabular reduction.

import numpy as np
from scipy.sparse import lil_matrix

from mdpforge.core.model import GenericModel


class Model(GenericModel):
    """Hexagonal-grid soccer as an ordered-discrete tabular control problem.

    State: (player_cell, ball_cell, defender_cell).  Actions 0..5 move/kick in
    hex directions.  The defender is deterministic and greedily moves toward the
    ball, yielding a compact single-agent MDP suitable for tabular comparisons.
    """

    HEX_DIRS = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]

    def __init__(self, state_dim: int, action_dim: int):
        target = max(27, state_dim)
        radius = 1
        while (1 + 3 * radius * (radius + 1)) ** 3 <= target and radius < 4:
            radius += 1
        self.radius = max(1, radius - 1)
        self.cells = self._build_cells(self.radius)
        self.cell_to_idx = {c: i for i, c in enumerate(self.cells)}
        self.n_cells = len(self.cells)
        self.action_dim = 6
        self.state_dim = self.n_cells**3
        self.name = f"{self.state_dim}_{self.action_dim}_hexagonal_grid_soccer"

    def _build_model(self):
        self.left_goal = min(range(self.n_cells), key=lambda k: self.cells[k][0])
        self.right_goal = max(range(self.n_cells), key=lambda k: self.cells[k][0])
        self.reward_matrix = -np.ones((self.state_dim, self.action_dim), dtype=float)
        self.transition_matrix = [
            lil_matrix((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]

        for s in range(self.state_dim):
            player, ball, defender = self._decode(s)
            if ball == self.right_goal:
                for a in range(self.action_dim):
                    self.transition_matrix[a][s, s] = 1.0
                    self.reward_matrix[s, a] = 0.0
                continue
            for a in range(self.action_dim):
                ns_tuple, reward = self._step(player, ball, defender, a)
                ns = self._encode(*ns_tuple)
                self.transition_matrix[a][s, ns] = 1.0
                self.reward_matrix[s, a] = reward
        self.transition_matrix = [m.tocsr() for m in self.transition_matrix]

    def _build_cells(self, r):
        cells = []
        for q in range(-r, r + 1):
            for rr in range(-r, r + 1):
                if -r <= -q - rr <= r:
                    cells.append((q, rr))
        return cells

    def _encode(self, player, ball, defender):
        return (player * self.n_cells + ball) * self.n_cells + defender

    def _decode(self, s):
        defender = s % self.n_cells
        s //= self.n_cells
        ball = s % self.n_cells
        s //= self.n_cells
        return s, ball, defender

    def _neighbor(self, idx, action):
        q, r = self.cells[idx]
        dq, dr = self.HEX_DIRS[action]
        return self.cell_to_idx.get((q + dq, r + dr), idx)

    def _dist(self, a, b):
        q1, r1 = self.cells[a]
        q2, r2 = self.cells[b]
        return max(abs(q1 - q2), abs(r1 - r2), abs((-q1 - r1) - (-q2 - r2)))

    def _defender_move(self, defender, ball, player):
        candidates = [defender] + [self._neighbor(defender, a) for a in range(6)]
        candidates = [c for c in candidates if c != player]
        return min(candidates, key=lambda c: self._dist(c, ball))

    def _step(self, player, ball, defender, action):
        reward = -1.0
        target = self._neighbor(player, action)
        if target == defender:
            target = player
        if player == ball:
            ball_target = self._neighbor(ball, action)
            if ball_target != defender:
                ball = ball_target
        player = target
        defender = self._defender_move(defender, ball, player)
        if defender == ball:
            reward = -25.0
            ball = self.left_goal
        if ball == self.right_goal:
            reward = 100.0
        return (player, ball, defender), reward
