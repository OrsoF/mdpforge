import numpy as np
from scipy.sparse import lil_matrix

from mdpforge.core.mdp import MDP

METADATA = {
    "tags": ["navigation", "grid-world"],
    "sizes": {
        "small": {
            "state_dim": 100,
            "parameters": {"state_dim": 100},
            "source": "measured",
        },
        "medium": {
            "state_dim": 1023,
            "parameters": {"state_dim": 1000},
            "source": "measured",
        },
        "large": {
            "state_dim": 7482,
            "parameters": {"state_dim": 7451},
            "source": "inferred",
        },
    },
    "category": "navigation",
    "description": "Grid navigation under column-dependent wind.",
    "reference": (
        "Sutton & Barto, Reinforcement Learning: An Introduction, Example 6.5, "
        "Windy Gridworld."
    ),
}


class Model(MDP):
    """
    Windy Gridworld finite MDP.

    Source:
        Sutton & Barto, Reinforcement Learning: An Introduction,
        Example 6.5, Windy Gridworld.

    State:
        s = row * width + col

    Actions:
        0 = North / Up
        1 = South / Down
        2 = East / Right
        3 = West / Left

    Dynamics:
        First apply the chosen action, then apply upward wind depending
        on the resulting column. Boundaries are clipped.

    Reward:
        -1 per transition until the goal.
        0 at the absorbing goal.
    """

    def __init__(self, state_dim: int = 70, action_dim: int = 10):
        # Classical Windy Gridworld is 7 x 10 = 70 states. Larger benchmark
        # requests use a square-ish grid, e.g. 100 -> 10 x 10 and 400 -> 20 x 20.
        state_dim = max(70, state_dim)

        if state_dim == 70:
            self.height = 7
            self.width = 10
        else:
            self.height = max(7, int(np.sqrt(state_dim)))
            self.width = max(10, int(np.ceil(state_dim / self.height)))
        self.state_dim = self.height * self.width

        self.action_dim = 4

        self.start_state = self._state(self.height // 2, 0)
        self.goal_state = self._state(
            self.height // 2, min(self.width - 3, self.width - 1)
        )

        self.wind = self._make_wind(self.width)

        self.name = "{}_{}_windy_gridworld".format(self.state_dim, self.action_dim)

    def _build_model(self):
        self.reward_matrix = -np.ones((self.state_dim, self.action_dim))

        self.transition_matrix = [
            lil_matrix((self.state_dim, self.state_dim)) for _ in range(self.action_dim)
        ]

        for s in range(self.state_dim):
            row, col = self._coord(s)

            for a in range(self.action_dim):
                if s == self.goal_state:
                    # Absorbing terminal state.
                    self.transition_matrix[a][s, s] = 1.0
                    self.reward_matrix[s, a] = 0.0
                    continue

                next_row, next_col = self._next_coord(row, col, a)
                next_state = self._state(next_row, next_col)

                self.transition_matrix[a][s, next_state] = 1.0

                if next_state == self.goal_state:
                    self.reward_matrix[s, a] = -1.0

        self.transition_matrix = [matrix.tocsr() for matrix in self.transition_matrix]

    def _make_wind(self, width: int):
        classical_wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

        if width <= len(classical_wind):
            return classical_wind[:width]

        # Extend with zero-wind columns on the right.
        return classical_wind + [0] * (width - len(classical_wind))

    def _coord(self, s: int):
        assert 0 <= s < self.state_dim
        return s // self.width, s % self.width

    def _state(self, row: int, col: int):
        return row * self.width + col

    def _next_coord(self, row: int, col: int, action: int):
        assert isinstance(action, int) and 0 <= action < 4

        if action == 0:  # North / Up
            row -= 1
        elif action == 1:  # South / Down
            row += 1
        elif action == 2:  # East / Right
            col += 1
        else:  # West / Left
            col -= 1

        # Clip after chosen action.
        row = min(max(row, 0), self.height - 1)
        col = min(max(col, 0), self.width - 1)

        # Apply upward wind from the resulting column.
        row -= self.wind[col]

        # Clip again after wind.
        row = min(max(row, 0), self.height - 1)

        return row, col
