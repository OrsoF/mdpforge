# Source: Buck, J. (2015). Mazes for Programmers: Code Your Own Twisty Little Passages.
import numpy as np
from mazelib import Maze
from mazelib.generate.BacktrackingGenerator import BacktrackingGenerator
from scipy.sparse import dok_array

from mdpforge.core.mdp import MDP

TEST_PARAMETERS = {"state_dim": 9, "action_dim": 4}

METADATA = {
    "sizes": {
        "tiny": {
            "requested_state_dim": 500,
            "parameters": {},
        },
        "small": {
            "requested_state_dim": 4000,
            "parameters": {},
        },
        "medium": {
            "requested_state_dim": 32000,
            "parameters": {},
        },
        "large": {
            "requested_state_dim": 256000,
            "parameters": {},
        },
    },
    "category": "navigation",
    "description": "Reach the absorbing exit in a backtracking maze with reward on entry.",
    "reference": (
        "Buck, J. (2015). Mazes for Programmers: Code Your Own Twisty Little Passages."
    ),
}


class Model(MDP):
    supported_criteria = frozenset({"total"})

    def __init__(self, state_dim: int = 100, action_dim: int = 10):
        self.requested_state_dim = int(state_dim)
        self.size = max(3, int(np.sqrt(state_dim)))
        self._build_maze()
        self._build_state_space()
        self.action_dim = 4

        self.name = "{}_{}_mazebacktrack_total".format(
            self.state_dim,
            self.action_dim,
        )
        self.terminal_states = (
            self.state_space[(2 * self.size - 1, 2 * self.size - 1)],
        )

    def _build_maze(self):
        maze = Maze()
        maze.generator = BacktrackingGenerator(self.size, self.size)
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
        """
        For given coords, return the next coord in the direction.
        """
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
        Initialize rewards to zero before marking actions that reach the exit.
        """
        self.reward_matrix = np.zeros(
            (self.state_dim, self.action_dim), dtype=np.float16
        )

    def _build_model(self):
        self._build_maze()
        self._build_reward_matrix()

        self.transition_matrix = [
            dok_array((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]

        exit_state = self.terminal_states[0]
        for ss1, (x, y) in enumerate(self.state_list):
            if ss1 == exit_state:
                # The exit stays absorbing in the total-reward version.
                for aa in range(self.action_dim):
                    self.transition_matrix[aa][ss1, ss1] = 1.0
                continue
            for aa in range(self.action_dim):
                new_x, new_y = self._update_coord(x, y, aa)
                ss2 = self.state_space[(new_x, new_y)]

                if not self.maze_grid[
                    (x + new_x) // 2, (y + new_y) // 2
                ]:  # Si ss2 est directement accessible
                    self.transition_matrix[aa][ss1, ss2] = 1.0
                else:
                    self.transition_matrix[aa][ss1, ss1] = 1.0

        self.transition_matrix = [matrix.tocsr() for matrix in self.transition_matrix]
        for action, transition in enumerate(self.transition_matrix):
            self.reward_matrix[:, action] = (
                transition[:, exit_state].toarray().ravel() > 0
            )
        self.reward_matrix[exit_state, :] = 0.0
