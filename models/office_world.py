# Source: Office World, adapted from Icarte et al. (2018) reward-machine benchmark;
# used as a tabular/factored benchmark in CAT+RL (Dadvar et al., 2023).

import numpy as np
from scipy.sparse import lil_matrix

from core.model import GenericModel


class Model(GenericModel):
    """Tabular Office World with position x {mail, coffee} flags.

    Actions: 0=N, 1=S, 2=E, 3=W.
    State: (cell, has_mail, has_coffee).  The office cell is absorbing once both
    items have been collected.  This is intentionally a compact reward-machine
    style abstraction rather than a pixel/world-object simulator.
    """

    def __init__(self, state_dim: int, action_dim: int):
        n_cells_target = max(25, int(np.ceil(max(1, state_dim) / 4.0)))
        self.height = max(5, int(np.sqrt(n_cells_target)))
        self.width = max(5, int(np.ceil(n_cells_target / self.height)))
        self.n_cells = self.height * self.width
        self.action_dim = 4
        self.state_dim = self.n_cells * 4
        self.name = f"{self.state_dim}_{self.action_dim}_office_world"

    def _build_model(self):
        self.wall_cells = self._default_walls()
        self.mail_cell = self._cell(1, 1)
        self.coffee_cell = self._cell(self.height - 2, 1)
        self.office_cell = self._cell(self.height - 2, self.width - 2)
        self.plant_cells = {self._cell(1, self.width - 2)}

        self.reward_matrix = -np.ones((self.state_dim, self.action_dim), dtype=float)
        self.transition_matrix = [
            lil_matrix((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]

        for s in range(self.state_dim):
            cell, has_mail, has_coffee = self._decode(s)
            if self._is_terminal(cell, has_mail, has_coffee):
                for a in range(self.action_dim):
                    self.transition_matrix[a][s, s] = 1.0
                    self.reward_matrix[s, a] = 0.0
                continue

            for a in range(self.action_dim):
                next_cell = self._move(cell, a)
                next_mail = has_mail or (next_cell == self.mail_cell)
                next_coffee = has_coffee or (next_cell == self.coffee_cell)
                ns = self._encode(next_cell, next_mail, next_coffee)
                self.transition_matrix[a][s, ns] = 1.0
                reward = -1.0
                if next_cell in self.plant_cells:
                    reward = -10.0
                if self._is_terminal(next_cell, next_mail, next_coffee):
                    reward = 50.0
                self.reward_matrix[s, a] = reward

        self.transition_matrix = [m.tocsr() for m in self.transition_matrix]

    def _cell(self, i: int, j: int) -> int:
        return i * self.width + j

    def _coord(self, cell: int):
        return cell // self.width, cell % self.width

    def _encode(self, cell: int, has_mail: bool, has_coffee: bool) -> int:
        return 4 * cell + 2 * int(has_mail) + int(has_coffee)

    def _decode(self, s: int):
        cell = s // 4
        flags = s % 4
        return cell, bool(flags & 2), bool(flags & 1)

    def _is_terminal(self, cell: int, has_mail: bool, has_coffee: bool) -> bool:
        return cell == self.office_cell and has_mail and has_coffee

    def _default_walls(self):
        walls = set()
        mid_col = self.width // 2
        mid_row = self.height // 2
        for i in range(self.height):
            if i not in {1, self.height - 2}:
                walls.add(self._cell(i, mid_col))
        for j in range(self.width):
            if j not in {1, self.width - 2}:
                walls.add(self._cell(mid_row, j))
        for c in (self.mail_cell if hasattr(self, "mail_cell") else None,):
            pass
        return walls

    def _move(self, cell: int, action: int) -> int:
        i, j = self._coord(cell)
        di_dj = [(-1, 0), (1, 0), (0, 1), (0, -1)][action]
        ni, nj = i + di_dj[0], j + di_dj[1]
        if not (0 <= ni < self.height and 0 <= nj < self.width):
            return cell
        nc = self._cell(ni, nj)
        return cell if nc in self.wall_cells else nc
