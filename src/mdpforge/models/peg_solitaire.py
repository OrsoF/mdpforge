# Source: Winston, W. L. and Goldberg, J. B. (2004). Operations Research: Applications and Algorithms, 4th ed.

import math

import numpy as np
from scipy.sparse import csr_matrix, identity

from mdpforge.core.mdp import MDP

TEST_PARAMETERS = {"state_dim": 512, "action_dim": 4}

METADATA = {
    "tags": ["games", "puzzle"],
    "sizes": {
        "small": {
            "state_dim": 128,
            "parameters": {"state_dim": 128},
            "source": "configured",
        },
        "medium": {
            "state_dim": 512,
            "parameters": {"state_dim": 512},
            "source": "configured",
        },
        "large": {
            "state_dim": 8192,
            "parameters": {"state_dim": 8192},
            "source": "configured",
        },
    },
    "category": "games",
    "description": (
        "Peg-jumping puzzle with board configurations represented as states."
    ),
    "reference": (
        "Winston, W. L. and Goldberg, J. B. (2004). Operations Research: "
        "Applications and Algorithms, 4th ed."
    ),
}


class Model(MDP):
    def __init__(self, state_dim: int = 512, action_dim: int = 10):
        if not isinstance(state_dim, (int, np.integer)) or state_dim < 1:
            raise ValueError("state_dim must be a positive integer")

        self.state_dim = int(state_dim)
        self.size = self._get_size(self.state_dim)
        self.action_dim = 4 * self.size * self.size
        self.name = "{}_{}_peg".format(self.state_dim, self.action_dim)

        # action_dim is retained for compatibility: an action specifies both
        # a starting cell and one of the four jump directions.

    def _get_size(self, state_dim: int) -> int:
        # Use enough cells to encode exactly state_dim indexed configurations.
        # A minimum 3x3 board is needed to make two-cell jumps possible.
        n_bits = (state_dim - 1).bit_length()
        size = math.isqrt(n_bits)
        return max(3, size + (size * size < n_bits))

    def _get_state(self, state_index: int) -> np.ndarray:
        if not 0 <= state_index < self.state_dim:
            raise IndexError("state index out of range")
        return np.array(
            list(format(state_index, "0{}b".format(self.size * self.size))),
            dtype=int,
        ).reshape(self.size, self.size)

    def _get_state_index(self, state: np.ndarray) -> int:
        return int("".join(map(str, state.flatten())), 2)

    def _get_action(self, action_index: int) -> tuple:
        direction = action_index // (self.size * self.size)
        position = action_index % (self.size * self.size)
        row = position // self.size
        col = position % self.size
        return (direction, row, col)

    def _build_model(self):
        n_states = self.state_dim
        n_cells = self.size * self.size
        states = np.arange(n_states, dtype=np.int64)
        rows = np.arange(n_states, dtype=np.int32)
        indptr = np.arange(n_states + 1, dtype=np.int32)
        data = np.ones(n_states, dtype=float)
        identity_matrix = identity(n_states, format="csr", dtype=float)

        # The same CSR data/indptr arrays are shared across deterministic
        # transitions. Only the successor indices depend on the action.
        self.transition_matrix = []
        directions = ((-1, 0), (0, 1), (1, 0), (0, -1))

        for action_index in range(self.action_dim):
            direction, row, col = self._get_action(action_index)
            dr, dc = directions[direction]
            middle_row, middle_col = row + dr, col + dc
            target_row, target_col = row + 2 * dr, col + 2 * dc

            if not (0 <= target_row < self.size and 0 <= target_col < self.size):
                self.transition_matrix.append(identity_matrix)
                continue

            start = n_cells - 1 - (row * self.size + col)
            middle = n_cells - 1 - (middle_row * self.size + middle_col)
            target = n_cells - 1 - (target_row * self.size + target_col)

            occupied = (1 << start) | (1 << middle)
            destination = 1 << target
            jump_mask = occupied | destination

            valid = ((states & occupied) == occupied) & ((states & destination) == 0)
            if not np.any(valid):
                self.transition_matrix.append(identity_matrix)
                continue

            # For arbitrary state_dim, the indexed states are a subset of all
            # board configurations. Moves leaving that subset are self-loops.
            valid_rows = np.flatnonzero(valid)
            successors = states[valid_rows] ^ jump_mask
            inside = successors < n_states
            if not np.any(inside):
                self.transition_matrix.append(identity_matrix)
                continue

            next_states = rows.copy()
            next_states[valid_rows[inside]] = successors[inside]
            self.transition_matrix.append(
                csr_matrix(
                    (data, next_states, indptr),
                    shape=(n_states, n_states),
                    copy=False,
                )
            )

        # Keep the original goal (a single peg at [2, 2]). Broadcasting
        # avoids allocating an otherwise mostly-zero state-action reward array.
        final_state = np.zeros((self.size, self.size), dtype=int)
        final_state[2, 2] = 1
        final_state_index = self._get_state_index(final_state)

        rewards = np.zeros(n_states, dtype=float)
        if final_state_index < n_states:
            rewards[final_state_index] = 1.0
        self.reward_matrix = np.broadcast_to(rewards[:, None], (n_states, self.action_dim))

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
