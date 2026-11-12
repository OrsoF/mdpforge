"""Finite-return maze_backtrack model with a one-percent per-step timeout."""

from mdpforge.models.discounted.maze_backtrack import Model as SourceModel
from mdpforge.models.total._episodic import EpisodicModel

METADATA = {
    "category": "navigation",
    "description": "Reach the exit in a backtracking maze with a 1% step timeout.",
    "reference": "Buck, J. (2015). Mazes for Programmers.",
    "sizes": {
        "small": {"requested_state_dim": 4000, "parameters": {"seed": 0}},
        "medium": {"requested_state_dim": 32000, "parameters": {"seed": 0}},
        "large": {"requested_state_dim": 256000, "parameters": {"seed": 0}},
    },
}

TEST_PARAMETERS = {"state_dim": 9, "seed": 0}


class Model(EpisodicModel):
    """Reach the exit in a backtracking maze with episodic termination."""

    source_model = SourceModel

    def __init__(self, state_dim: int = 100, *, seed: int = 0):
        super().__init__(state_dim, seed=seed)

    def _goal_states(self, source):
        return (source.state_space[(2 * source.size - 1, 2 * source.size - 1)],)
