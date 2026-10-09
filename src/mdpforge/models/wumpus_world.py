# Source: Wumpus World, Russell & Norvig, Artificial Intelligence: A Modern Approach;
# used as a discrete-state benchmark in CAT+RL (Dadvar et al., 2023).

import numpy as np
from scipy.sparse import lil_matrix

from mdpforge.core.mdp import MDP


class Model(MDP):
    """Classic 4x4 Wumpus World as a fully enumerated tabular MDP.

    Actions: 0=forward, 1=turn_left, 2=turn_right, 3=grab, 4=shoot, 5=climb.
    State: (cell, direction, has_gold, wumpus_alive, has_arrow, terminal).
    """

    def __init__(self, state_dim: int = 1024, action_dim: int = 10):
        self.size = max(4, int(round(np.sqrt(max(16, state_dim // 32)))))
        self.n_cells = self.size * self.size
        self.action_dim = 6
        self.state_dim = self.n_cells * 4 * 2 * 2 * 2 * 2
        self.name = f"{self.state_dim}_{self.action_dim}_wumpus_world"

    def _build_model(self):
        self.start_cell = self._cell(self.size - 1, 0)
        self.gold_cell = self._cell(0, self.size - 1)
        self.wumpus_cell = self._cell(1, self.size - 1)
        self.pit_cells = {
            self._cell(2, 2),
            self._cell(3, 3) if self.size > 3 else self._cell(0, 2),
        }
        self.pit_cells.discard(self.start_cell)
        self.pit_cells.discard(self.gold_cell)
        self.pit_cells.discard(self.wumpus_cell)

        self.reward_matrix = -np.ones((self.state_dim, self.action_dim), dtype=float)
        self.transition_matrix = [
            lil_matrix((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]

        for s in range(self.state_dim):
            cell, direction, has_gold, wumpus_alive, has_arrow, terminal = self._decode(
                s
            )
            if terminal:
                for a in range(self.action_dim):
                    self.transition_matrix[a][s, s] = 1.0
                    self.reward_matrix[s, a] = 0.0
                continue

            for a in range(self.action_dim):
                ns_tuple, reward = self._step(
                    cell, direction, has_gold, wumpus_alive, has_arrow, a
                )
                ns = self._encode(*ns_tuple)
                self.transition_matrix[a][s, ns] = 1.0
                self.reward_matrix[s, a] = reward

        self.transition_matrix = [m.tocsr() for m in self.transition_matrix]

    def _cell(self, i: int, j: int) -> int:
        return i * self.size + j

    def _coord(self, cell: int):
        return cell // self.size, cell % self.size

    def _encode(
        self,
        cell: int,
        direction: int,
        has_gold: bool,
        wumpus_alive: bool,
        has_arrow: bool,
        terminal: bool,
    ) -> int:
        x = cell
        x = x * 4 + direction
        x = x * 2 + int(has_gold)
        x = x * 2 + int(wumpus_alive)
        x = x * 2 + int(has_arrow)
        x = x * 2 + int(terminal)
        return x

    def _decode(self, s: int):
        terminal = bool(s % 2)
        s //= 2
        has_arrow = bool(s % 2)
        s //= 2
        wumpus_alive = bool(s % 2)
        s //= 2
        has_gold = bool(s % 2)
        s //= 2
        direction = s % 4
        s //= 4
        return s, direction, has_gold, wumpus_alive, has_arrow, terminal

    def _forward_cell(self, cell: int, direction: int) -> int:
        i, j = self._coord(cell)
        di, dj = [(-1, 0), (0, 1), (1, 0), (0, -1)][direction]
        ni, nj = i + di, j + dj
        if 0 <= ni < self.size and 0 <= nj < self.size:
            return self._cell(ni, nj)
        return cell

    def _arrow_hits_wumpus(self, cell: int, direction: int) -> bool:
        i, j = self._coord(cell)
        wi, wj = self._coord(self.wumpus_cell)
        return (
            (direction == 0 and j == wj and wi < i)
            or (direction == 1 and i == wi and wj > j)
            or (direction == 2 and j == wj and wi > i)
            or (direction == 3 and i == wi and wj < j)
        )

    def _step(self, cell, direction, has_gold, wumpus_alive, has_arrow, action):
        reward = -1.0
        terminal = False
        if action == 0:
            cell = self._forward_cell(cell, direction)
            if cell in self.pit_cells or (cell == self.wumpus_cell and wumpus_alive):
                reward = -100.0
                terminal = True
        elif action == 1:
            direction = (direction - 1) % 4
        elif action == 2:
            direction = (direction + 1) % 4
        elif action == 3:
            if cell == self.gold_cell and not has_gold:
                has_gold = True
                reward = 100.0
        elif action == 4:
            if has_arrow:
                has_arrow = False
                reward = -10.0
                if wumpus_alive and self._arrow_hits_wumpus(cell, direction):
                    wumpus_alive = False
                    reward = 50.0
        else:  # climb
            if cell == self.start_cell:
                terminal = True
                reward = 1000.0 if has_gold else -1.0
        return (cell, direction, has_gold, wumpus_alive, has_arrow, terminal), reward
