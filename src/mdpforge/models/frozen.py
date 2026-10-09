import numpy as np
from scipy.sparse import lil_matrix

from mdpforge.core.mdp import MDP


class Model(MDP):
    """
    FrozenLake finite MDP.

    Reference:
        Gymnasium FrozenLake-v1 / OpenAI Gym FrozenLake.

    State:
        s = row * side_size + col

    Actions:
        0 = Left
        1 = Down
        2 = Right
        3 = Up

    Rewards:
        +1 when entering the goal state, 0 otherwise.

    Terminal states:
        Holes and goal are absorbing.
    """

    def __init__(self, state_dim: int = 100, action_dim: int = 10):
        # Keep same style as other models: accept requested dimensions,
        # but enforce a square grid and action_dim = 4.
        side_size = int(np.sqrt(max(16, state_dim)))
        side_size = max(4, side_size)

        self.side_size = side_size
        self.state_dim = self.side_size**2
        self.action_dim = 4

        # Standard Gym FrozenLake 4x4 map if side_size == 4.
        # For larger grids, we generate a simple deterministic map:
        # S at top-left, G at bottom-right, sparse holes.
        self.desc = self._make_map(self.side_size)

        self.name = "{}_{}_frozen_lake".format(self.state_dim, self.action_dim)

    def _build_model(self):
        self.reward_matrix = np.zeros((self.state_dim, self.action_dim))

        self.transition_matrix = [
            lil_matrix((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]

        for s in range(self.state_dim):
            row, col = self._coord(s)
            cell = self.desc[row][col]

            for a in range(self.action_dim):
                if cell in ("H", "G"):
                    # Absorbing terminal states.
                    self.transition_matrix[a][s, s] = 1.0
                    self.reward_matrix[s, a] = 0.0
                    continue

                # Slippery FrozenLake:
                # intended action plus two perpendicular actions.
                for executed_action in self._slip_actions(a):
                    next_row, next_col = self._next_coord(row, col, executed_action)
                    next_state = self._state(next_row, next_col)
                    next_cell = self.desc[next_row][next_col]

                    self.transition_matrix[a][s, next_state] += 1.0 / 3.0

                    if next_cell == "G":
                        # reward_matrix in this codebase is R[s, a],
                        # so we store expected immediate reward.
                        self.reward_matrix[s, a] += 1.0 / 3.0

        self.transition_matrix = [matrix.tocsr() for matrix in self.transition_matrix]

    def _make_map(self, side_size: int):
        if side_size == 4:
            return [
                "SFFF",
                "FHFH",
                "FFFH",
                "HFFG",
            ]

        grid = [["F" for _ in range(side_size)] for _ in range(side_size)]
        grid[0][0] = "S"
        grid[side_size - 1][side_size - 1] = "G"

        # Parameterized sparse-hole pattern.
        # Avoid start, goal, and a simple right-then-down safe corridor.
        for i in range(side_size):
            for j in range(side_size):
                if (i, j) in [(0, 0), (side_size - 1, side_size - 1)]:
                    continue
                if i == 0 or j == side_size - 1:
                    continue
                if (3 * i + 5 * j) % 11 == 0:
                    grid[i][j] = "H"

        return ["".join(row) for row in grid]

    def _coord(self, s: int):
        return s // self.side_size, s % self.side_size

    def _state(self, row: int, col: int):
        return row * self.side_size + col

    def _next_coord(self, row: int, col: int, action: int):
        if action == 0:  # Left
            col -= 1
        elif action == 1:  # Down
            row += 1
        elif action == 2:  # Right
            col += 1
        elif action == 3:  # Up
            row -= 1
        else:
            raise ValueError("Invalid action.")

        # Bouncy boundary: trying to leave the grid keeps the agent in place.
        row = min(max(row, 0), self.side_size - 1)
        col = min(max(col, 0), self.side_size - 1)

        return row, col

    def _slip_actions(self, action: int):
        """
        Gym-style slippery dynamics.

        If action is selected, the actual move is sampled uniformly from:
            left turn, intended action, right turn.

        With action order:
            0 = Left
            1 = Down
            2 = Right
            3 = Up

        This is exactly:
            (action - 1) % 4, action, (action + 1) % 4
        """
        return [(action - 1) % 4, action, (action + 1) % 4]
