# Source:
# Prim, R. C. (1957).
# Shortest connection networks and some generalizations.
# Bell System Technical Journal 36(6): 1389–1401.

import numpy as np
from mazelib import Maze
from mazelib.generate.Prims import Prims
from scipy.sparse import dok_array

from core.model import GenericModel


class Model(GenericModel):
    def __init__(self, state_dim: int, action_dim: int):
        del action_dim  # This environment always has four actions.

        if state_dim <= 0:
            raise ValueError("state_dim must be positive.")

        self.size = int(np.sqrt(state_dim))
        state_dim = self.size**2

        self.action_dim = 4

        self._build_maze()
        self._build_state_space()

        self.name = f"{self.state_dim}_{self.action_dim}_mazeprims_total"

    def _build_model(self) -> None:
        self._build_transition_matrix()
        self._build_reward_matrix()

    def _build_maze(self) -> None:
        maze = Maze()
        maze.generator = Prims(self.size, self.size)
        maze.generate()

        maze.start = (1, 1)
        maze.end = (
            2 * self.size - 1,
            2 * self.size - 1,
        )

        self.start_coord = maze.start
        self.exit_coord = maze.end
        self.maze_grid = maze.grid

    def _build_state_space(self) -> None:
        self.state_list = [
            (x, y)
            for x in range(1, 2 * self.size, 2)
            for y in range(1, 2 * self.size, 2)
            if self.maze_grid[x, y] == 0
        ]

        self.state_space = {
            coord: state_index for state_index, coord in enumerate(self.state_list)
        }

        self.state_dim = len(self.state_list)

        if self.start_coord not in self.state_space:
            raise ValueError(
                f"Start coordinate {self.start_coord} is not a valid maze state."
            )

        if self.exit_coord not in self.state_space:
            raise ValueError(
                f"Exit coordinate {self.exit_coord} is not a valid maze state."
            )

    def _update_coord(
        self,
        x: int,
        y: int,
        direction: int,
    ) -> tuple[int, int]:
        if direction == 0:
            new_x, new_y = x + 2, y
        elif direction == 1:
            new_x, new_y = x - 2, y
        elif direction == 2:
            new_x, new_y = x, y + 2
        elif direction == 3:
            new_x, new_y = x, y - 2
        else:
            raise ValueError(f"Invalid action: {direction}")

        new_x = int(np.clip(new_x, 1, 2 * self.size - 1))
        new_y = int(np.clip(new_y, 1, 2 * self.size - 1))

        return new_x, new_y

    def _is_accessible(
        self,
        x: int,
        y: int,
        new_x: int,
        new_y: int,
    ) -> bool:
        wall_x = (x + new_x) // 2
        wall_y = (y + new_y) // 2

        return self.maze_grid[wall_x, wall_y] == 0

    def _build_transition_matrix(self) -> None:
        self.transition_matrix = [
            dok_array(
                (self.state_dim, self.state_dim),
                dtype=np.float64,
            )
            for _ in range(self.action_dim)
        ]

        for state, (x, y) in enumerate(self.state_list):
            for action in range(self.action_dim):
                new_x, new_y = self._update_coord(x, y, action)

                if self._is_accessible(x, y, new_x, new_y):
                    next_state = self.state_space[(new_x, new_y)]
                else:
                    next_state = state

                self.transition_matrix[action][state, next_state] = 1.0

        self.transition_matrix = [matrix.tocsr() for matrix in self.transition_matrix]

        for action, matrix in enumerate(self.transition_matrix):
            row_sums = np.asarray(matrix.sum(axis=1)).ravel()

            if not np.allclose(row_sums, 1.0, atol=1e-12, rtol=0.0):
                invalid_rows = np.flatnonzero(
                    ~np.isclose(row_sums, 1.0, atol=1e-12, rtol=0.0)
                )
                raise ValueError(
                    f"Transition matrix for action {action} is not stochastic. "
                    f"Invalid rows: {invalid_rows.tolist()}; "
                    f"sums: {row_sums[invalid_rows].tolist()}."
                )

    def _build_reward_matrix(self) -> None:
        """
        Give reward 1 when an action transitions into the exit state.
        All other rewards are zero.
        """
        self.reward_matrix = np.zeros(
            (self.state_dim, self.action_dim),
            dtype=np.float64,
        )

        exit_state = self.state_space[self.exit_coord]

        for state in range(self.state_dim):
            if state == exit_state:
                continue

            for action in range(self.action_dim):
                if self.transition_matrix[action][state, exit_state] > 0:
                    self.reward_matrix[state, action] = 1.0

        if not np.any(self.reward_matrix > 0):
            raise ValueError("No transition reaches the maze exit.")
