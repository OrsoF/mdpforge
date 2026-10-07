# Source: Wilson, D. B. (1996). Generating random spanning trees more quickly than the cover time. STOC, pp. 296-303.
import numpy as np
from mazelib import Maze
from mazelib.generate.Wilsons import Wilsons
from scipy.sparse import dok_array

from core.model import GenericModel


class Model(GenericModel):
    def __init__(self, state_dim: int, action_dim: int):
        self.size = int(np.sqrt(state_dim))
        self._build_maze()
        self._build_state_space()
        self.action_dim = 4

        self.name = "{}_{}_mazewilson_total".format(
            self.state_dim,
            self.action_dim,
        )

    def _build_maze(self):
        maze = Maze()
        maze.generator = Wilsons(self.size, self.size)
        maze.generate()  # genere le labyrinthe mais pas les entree à nous de les choisirs
        maze.start = (1, 1)
        maze.end = (2 * self.size - 1, 2 * self.size - 1)
        self.maze_grid = maze.grid

    def _build_state_space(self):
        self.state_list = []
        for x in range(1, 2 * self.size + 1, 2):
            for y in range(1, 2 * self.size + 1, 2):
                if not self.maze_grid[x, y]:  # We are in the maze
                    self.state_list.append((x, y))
        self.state_space = {
            coord: state_index for (state_index, coord) in enumerate(self.state_list)
        }
        self.state_dim = len(self.state_list)

    def _update_coord(self, x: int, y: int, direction: int) -> tuple:
        if direction == 0:
            x += 2
        elif direction == 1:
            x -= 2
        elif direction == 2:
            y += 2
        else:
            y -= 2
        x = np.clip(x, 1, 2 * self.size - 1)
        y = np.clip(y, 1, 2 * self.size - 1)
        return x, y

    def _build_reward_matrix(self):
        """
        The reward is always -1, except when reaching the exit.
        """
        # self.reward_matrix = -1 * np.ones((self.state_dim, self.action_dim))
        # self.reward_matrix[self.state_space[(2 * self.size - 1, 2 * self.size - 1)]] = (
        #     0.0
        # )

        self.reward_matrix = np.zeros((self.state_dim, self.action_dim))

        if self.transition_matrix[2][self.state_dim - 2, self.state_dim - 1] > 0:
            self.reward_matrix[self.state_dim - 2, 2] = 1.0
        if (
            self.transition_matrix[1][
                self.state_dim - self.size - 1, self.state_dim - 1
            ]
            > 0
        ):
            self.reward_matrix[self.state_dim - 2, 2] = 1.0

        assert np.sum(self.reward_matrix) > 0

    def _build_model(self):
        self._build_maze()

        self.transition_matrix = [
            dok_array((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]

        for ss1, (x, y) in enumerate(self.state_list):
            for aa in range(self.action_dim):
                new_x, new_y = self._update_coord(x, y, aa)
                ss2 = self.state_space[(new_x, new_y)]

                if not self.maze_grid[
                    (x + new_x) // 2, (y + new_y) // 2
                ]:  # Si ss2 est directement accessible
                    self.transition_matrix[aa][ss1, ss2] = 1.0
                else:
                    self.transition_matrix[aa][ss1, ss1] = 1.0

        self._build_reward_matrix()

        self.transition_matrix = [matrix.tocsr() for matrix in self.transition_matrix]
