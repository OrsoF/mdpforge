# Source: inspired by Hengst, B. (2012). Hierarchical approaches.
# In Reinforcement Learning: State-of-the-Art, pp. 293-323.
#
# 9 Rooms variant:
# - 3 x 3 rooms
# - stochastic movements: 0.8 intended move, 0.2 stay
# - internal walls with doors
# - absorbing exit state with reward 0

import numpy as np
from scipy.sparse import lil_matrix

from mdpforge.core.mdp import MDP

METADATA = {
    "category": "navigation",
    "description": "Navigation through nine connected rooms with uncertain movement.",
    "reference": (
        "Inspired by Hengst, B. (2012). Hierarchical approaches. In Reinforcement "
        "Learning: State-of-the-Art, pp. 293-323."
    ),
}


class Model(MDP):
    def __init__(self, state_dim: int = 81, action_dim: int = 10):
        state_dim = max(81, state_dim)

        # For 9 rooms, total side length = 3 * size_of_one_room.
        self.size_of_one_room = max(1, int(np.sqrt(state_dim) / 3.0))
        self.n_doors = 1

        self.action_dim = 4
        self.side_size = 3 * self.size_of_one_room
        self.state_dim = self.side_size**2

        self.name = "{}_{}_9_rooms".format(self.state_dim, self.action_dim)

    def _build_model(self):
        self.door_step = self.size_of_one_room / (self.n_doors + 1)

        self.reward_matrix = -np.ones((self.state_dim, self.action_dim))
        self.transition_matrix: list = [
            lil_matrix((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]

        self.transition_matrix = self._add_displacements(self.transition_matrix)
        self.transition_matrix = self._add_walls(self.transition_matrix)
        self.transition_matrix, self.reward_matrix = self._add_exit(
            self.transition_matrix, self.reward_matrix
        )

        self.transition_matrix = [matrix.tocsr() for matrix in self.transition_matrix]

    def _coord(self, ss: int):
        assert ss < self.side_size**2
        return ss // self.side_size, ss % self.side_size

    def _state(self, i: int, j: int):
        return i * self.side_size + j

    def _next_state(self, i: int, j: int, direction: int):
        assert isinstance(direction, int) and 0 <= direction < 4

        if direction == 0:  # North
            return i - 1, j
        elif direction == 1:  # South
            return i + 1, j
        elif direction == 2:  # East
            return i, j + 1
        else:  # West
            return i, j - 1

    def _is_in_grid(self, i: int, j: int):
        return 0 <= i < self.side_size and 0 <= j < self.side_size

    def _add_displacements(self, transition_matrix: list):
        for a in range(self.action_dim):
            for ss in range(self.state_dim):
                i, j = self._coord(ss)
                next_coord = self._next_state(i, j, a)

                if self._is_in_grid(*next_coord):
                    transition_matrix[a][ss, ss] = 0.2
                    transition_matrix[a][ss, self._state(*next_coord)] = 0.8
                else:
                    transition_matrix[a][ss, ss] = 1.0

        return transition_matrix

    def _add_wall_element(
        self,
        ss1: int,
        ss2: int,
        transition_matrix: list,
        direction: str,
    ):
        assert direction in ["horizontal", "vertical"]

        i, j = self._coord(ss1)
        k, l = self._coord(ss2)

        assert (i - k) ** 2 + (j - l) ** 2 == 1

        if direction == "vertical":
            assert j < l

            # East from ss1 blocked.
            transition_matrix[2][ss1, ss1] = 1.0
            transition_matrix[2][ss1, ss2] = 0.0

            # West from ss2 blocked.
            transition_matrix[3][ss2, ss2] = 1.0
            transition_matrix[3][ss2, ss1] = 0.0

        else:
            assert i < k

            # South from ss1 blocked.
            transition_matrix[1][ss1, ss1] = 1.0
            transition_matrix[1][ss1, ss2] = 0.0

            # North from ss2 blocked.
            transition_matrix[0][ss2, ss2] = 1.0
            transition_matrix[0][ss2, ss1] = 0.0

        return transition_matrix

    def _door_indices_for_wall_segment(self, segment_start: int):
        """
        Doors along one room-sized wall segment.

        Example with n_doors = 1:
            the door is placed roughly at the middle of each segment.
        """
        return {
            segment_start + int(i * self.door_step) for i in range(1, self.n_doors + 1)
        }

    def _add_horizontal_wall(self, transition_matrix: list, wall_row: int):
        """
        Add a horizontal wall between rows wall_row and wall_row + 1.

        The wall crosses the full grid width, but each room-to-room segment
        has its own doors.
        """
        direction = "horizontal"
        i, k = wall_row, wall_row + 1

        doors = set()
        for room_col in range(3):
            segment_start = room_col * self.size_of_one_room
            doors |= self._door_indices_for_wall_segment(segment_start)

        for col in range(self.side_size):
            if col not in doors:
                ss1 = self._state(i, col)
                ss2 = self._state(k, col)
                transition_matrix = self._add_wall_element(
                    ss1,
                    ss2,
                    transition_matrix,
                    direction,
                )

        return transition_matrix

    def _add_vertical_wall(self, transition_matrix: list, wall_col: int):
        """
        Add a vertical wall between cols wall_col and wall_col + 1.

        The wall crosses the full grid height, but each room-to-room segment
        has its own doors.
        """
        direction = "vertical"
        j, l = wall_col, wall_col + 1

        doors = set()
        for room_row in range(3):
            segment_start = room_row * self.size_of_one_room
            doors |= self._door_indices_for_wall_segment(segment_start)

        for row in range(self.side_size):
            if row not in doors:
                ss1 = self._state(row, j)
                ss2 = self._state(row, l)
                transition_matrix = self._add_wall_element(
                    ss1,
                    ss2,
                    transition_matrix,
                    direction,
                )

        return transition_matrix

    def _add_walls(self, transition_matrix: list):
        # Horizontal walls between room rows:
        # row block 0 | row block 1 | row block 2
        for cut in [self.size_of_one_room, 2 * self.size_of_one_room]:
            wall_row = cut - 1
            transition_matrix = self._add_horizontal_wall(
                transition_matrix,
                wall_row,
            )

        # Vertical walls between room columns:
        # col block 0 | col block 1 | col block 2
        for cut in [self.size_of_one_room, 2 * self.size_of_one_room]:
            wall_col = cut - 1
            transition_matrix = self._add_vertical_wall(
                transition_matrix,
                wall_col,
            )

        return transition_matrix

    def _add_exit(self, transition_matrix: list, reward_matrix: np.ndarray):
        """
        Add absorbing exit in the upper-left room.

        Analogous to the Four Rooms implementation:
        i = 0, j = side_size // 6 is inside the first room.
        """
        i, j = 0, max(0, self.side_size // 6)
        exit_state = self._state(i, j)

        for a in range(self.action_dim):
            transition_matrix[a][exit_state, :] = 0
            transition_matrix[a][exit_state, exit_state] = 1.0

        reward_matrix[exit_state, :] = 0.0

        return transition_matrix, reward_matrix
