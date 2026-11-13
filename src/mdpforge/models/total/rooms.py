# Source: Hengst, B. (2012). Hierarchical approaches. In Reinforcement Learning: State-of-the-Art, pp. 293-323.

import numpy as np
from scipy.sparse import lil_matrix

from mdpforge.core.mdp import MDP

DIRECTIONS = ((-1, 0), (1, 0), (0, 1), (0, -1))  # North, south, east, west

METADATA = {
    "sizes": {
        "tiny": {
            "requested_state_dim": 1000,
            "parameters": {},
        },
        "small": {
            "requested_state_dim": 10000,
            "parameters": {},
        },
        "medium": {
            "requested_state_dim": 40000,
            "parameters": {},
        },
        "large": {
            "requested_state_dim": 160000,
            "parameters": {},
        },
    },
    "category": "navigation",
    "description": "Reach the absorbing exit through four rooms with reward on entry.",
    "reference": (
        "Hengst, B. (2012). Hierarchical approaches. In Reinforcement Learning: "
        "State-of-the-Art, pp. 293-323."
    ),
}


TEST_PARAMETERS = {"state_dim": 100}


class Model(MDP):
    supported_criteria = frozenset({"total"})

    def __init__(self, state_dim: int = 100, action_dim: int = 10):
        self.requested_state_dim = int(state_dim)
        state_dim = max(100, state_dim)
        self.size_of_one_room = int(np.sqrt(state_dim) / 2.0)
        self.n_doors = 1

        self.action_dim = 4
        self.state_dim = 4 * self.size_of_one_room**2

        self.name = "{}_{}_rooms_total".format(
            self.state_dim, self.action_dim
        )
        self.terminal_states = (self.size_of_one_room // 2,)

    def _build_model(self):
        self.side_size = self.size_of_one_room * 2

        self.door_step = self.size_of_one_room / (self.n_doors + 1)

        self.doors_indices = [
            int(i * self.door_step) for i in range(1, self.n_doors + 1)
        ] + [
            self.size_of_one_room + int(i * self.door_step)
            for i in range(1, self.n_doors + 1)
        ]

        self.reward_matrix = np.zeros((self.state_dim, self.action_dim))
        self.transition_matrix: list = [
            lil_matrix((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]
        self._add_displacements()
        self._add_walls()
        self._add_exit()

        self.transition_matrix = [matrix.tocsr() for matrix in self.transition_matrix]
        exit_state = self.terminal_states[0]
        for action, transition in enumerate(self.transition_matrix):
            self.reward_matrix[:, action] = (
                transition[:, exit_state].toarray().ravel() > 0
            )
        self.reward_matrix[exit_state, :] = 0.0

    def _coord(self, ss: int):
        return ss // self.side_size, ss % self.side_size

    def _state(self, i: int, j: int):
        return i * self.side_size + j

    def _is_in_grid(self, i: int, j: int):
        return 0 <= i < self.side_size and 0 <= j < self.side_size

    def _add_displacements(self):
        for action, (row_step, col_step) in enumerate(DIRECTIONS):
            for ss in range(self.state_dim):
                i, j = self._coord(ss)
                next_coord = i + row_step, j + col_step
                if self._is_in_grid(*next_coord):
                    self.transition_matrix[action][ss, ss] = 0.2
                    self.transition_matrix[action][ss, self._state(*next_coord)] = 0.8
                else:
                    self.transition_matrix[action][ss, ss] = 1.0

    def _block_edge(self, first: int, second: int, forward: int, backward: int):
        self.transition_matrix[forward][first, first] = 1.0
        self.transition_matrix[forward][first, second] = 0.0
        self.transition_matrix[backward][second, second] = 1.0
        self.transition_matrix[backward][second, first] = 0.0

    def _add_walls(self):
        middle = self.side_size // 2
        for index in range(self.side_size):
            if index in self.doors_indices:
                continue

            upper = self._state(middle - 1, index)
            lower = self._state(middle, index)
            self._block_edge(upper, lower, 1, 0)  # South, north

            left = self._state(index, middle - 1)
            right = self._state(index, middle)
            self._block_edge(left, right, 2, 3)  # East, west

    def _add_exit(self):
        """Add exit at the third top left box."""
        i, j = 0, self.side_size // 4
        exit_state = self._state(i, j)
        for a in range(self.action_dim):
            self.transition_matrix[a][exit_state, :] = 0
            self.transition_matrix[a][exit_state, exit_state] = 1
        self.reward_matrix[exit_state, :] = 0
