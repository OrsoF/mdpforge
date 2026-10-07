# Source: Winston, W. L. and Goldberg, J. B. (2004). Operations Research: Applications and Algorithms, 4th ed.

import numpy as np
from scipy.sparse import lil_matrix

from core.model import GenericModel


class Model(GenericModel):
    def __init__(self, state_dim: int, action_dim: int):
        self.size = self._get_size(state_dim)

        self.state_dim = (
            2 ** (self.size * self.size) - 1
        )  # 24 states (5x5 grid with one peg missing)
        self.action_dim = 4 * self.size * self.size  # 4 directions for each cell

        self.name = "{}_{}_peg".format(state_dim, action_dim)

    def _get_size(self, state_dim: int) -> int:
        return int(np.round(np.sqrt(np.log2(state_dim)), 0))

    def _get_state(self, state_index: int) -> np.ndarray:
        return np.array(
            list(bin(state_index)[2:].zfill(self.size * self.size)), dtype=int
        ).reshape(self.size, self.size)

    def _get_state_index(self, state: np.ndarray):
        return int("".join(map(str, state.flatten())), 2)

    def _get_action(self, action_index: int) -> tuple:
        direction = action_index // (self.size * self.size)
        position = action_index % (self.size * self.size)
        row = position // self.size
        col = position % self.size
        return (direction, row, col)

    def _build_model(self):
        size = self.size
        self.state_dim = (
            2 ** (size * size) - 1
        )  # 24 states (5x5 grid with one peg missing)
        self.action_dim = 4 * size * size  # 4 directions for each cell

        self.transition_matrix = [
            lil_matrix((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]
        self.reward_matrix = np.zeros((self.state_dim, self.action_dim))

        for state_index in range(self.state_dim):
            state = self._get_state(state_index)
            for action_index in range(self.action_dim):
                direction, row, col = self._get_action(action_index)

                if not self._is_action_valid(state, direction, row, col):
                    self.transition_matrix[action_index][state_index, state_index] = 1.0

                else:
                    state[row, col] = 0
                    if direction == 0:
                        state[row - 1, col] = 0
                        state[row - 2, col] = 1
                    elif direction == 1:
                        state[row, col + 1] = 0
                        state[row, col + 2] = 1
                    elif direction == 2:
                        state[row + 1, col] = 0
                        state[row + 2, col] = 1
                    elif direction == 3:
                        state[row, col - 1] = 0
                        state[row, col - 2] = 1

                    new_state_index = int("".join(map(str, state.flatten())), 2)
                    self.transition_matrix[action_index][
                        state_index, new_state_index
                    ] = 1.0

        final_state = np.zeros((self.size, self.size), dtype=int)
        final_state[2, 2] = 1
        final_state_index = int("".join(map(str, final_state.flatten())), 2)
        self.reward_matrix[final_state_index, :] = 1.0

        self.transition_matrix = [matrix.tocsr() for matrix in self.transition_matrix]

    def _is_action_valid(
        self, state: np.ndarray, direction: int, row: int, col: int
    ) -> bool:
        if state[row, col] == 0:
            return False
        if direction == 0:
            return row > 1 and state[row - 1, col] == 1 and state[row - 2, col] == 0
        elif direction == 1:
            return (
                col < self.size - 2
                and state[row, col + 1] == 1
                and state[row, col + 2] == 0
            )
        elif direction == 2:
            return (
                row < self.size - 2
                and state[row + 1, col] == 1
                and state[row + 2, col] == 0
            )
        elif direction == 3:
            return col > 1 and state[row, col - 1] == 1 and state[row, col - 2] == 0
        return False
