"""Finite-return m_maze model with a one-percent per-step timeout."""

from mdpforge.models.discounted.m_maze import Model as SourceModel
from mdpforge.models.total._episodic import EpisodicModel

METADATA = {
    "category": "navigation",
    "description": "Reach the goal in an M-shaped maze with episodic termination.",
    "reference": "McCallum (1996), U-Tree / UDM aliased maze variants.",
    "sizes": {
        "small": {"requested_state_dim": 8000, "parameters": {}},
        "medium": {"requested_state_dim": 64000, "parameters": {}},
        "large": {"requested_state_dim": 128000, "parameters": {}},
    },
}

TEST_PARAMETERS = {"state_dim": 81}


class Model(EpisodicModel):
    """Reach the goal in an M-shaped maze with episodic termination."""

    source_model = SourceModel

    def _goal_states(self, source):
        return (2 * source.width - 2,)
