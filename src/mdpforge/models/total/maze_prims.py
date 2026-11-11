"""Finite-return maze_prims model with a one-percent per-step timeout."""

from mdpforge.models.discounted.maze_prims import Model as SourceModel
from mdpforge.models.total._episodic import EpisodicModel

METADATA = {
    "category": "navigation",
    "description": "Reach the exit in a Prim maze with episodic termination.",
    "reference": (
        "Prim, R. C. (1957). Shortest connection networks and some "
        "generalizations. Bell System Technical Journal 36(6):1389-1401."
    ),
    "sizes": {
        "small": {"requested_state_dim": 12000, "parameters": {"seed": 0}},
        "medium": {"requested_state_dim": 48000, "parameters": {"seed": 0}},
        "large": {"requested_state_dim": 192000, "parameters": {"seed": 0}},
    },
}

TEST_PARAMETERS = {"state_dim": 9, "seed": 0}


class Model(EpisodicModel):
    """Reach the exit in a Prim maze with episodic termination."""

    source_model = SourceModel

    def __init__(self, state_dim: int = 100, *, seed: int = 0):
        super().__init__(state_dim, seed=seed)

    def _goal_states(self, source):
        return (source.state_space[(2 * source.size - 1, 2 * source.size - 1)],)
