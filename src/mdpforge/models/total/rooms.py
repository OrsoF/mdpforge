"""Finite-return rooms model with a one-percent per-step timeout."""

from mdpforge.models.discounted.rooms import Model as SourceModel
from mdpforge.models.total._episodic import EpisodicModel

METADATA = {
    "category": "navigation",
    "description": "Reach the exit through four connected rooms with a 1% step timeout.",
    "reference": "Hengst, B. (2012). Hierarchical approaches.",
    "sizes": {
        "small": {"requested_state_dim": 10000, "parameters": {}},
        "medium": {"requested_state_dim": 40000, "parameters": {}},
        "large": {"requested_state_dim": 160000, "parameters": {}},
    },
}

TEST_PARAMETERS = {"state_dim": 100}


class Model(EpisodicModel):
    """Reach the exit through four connected rooms with episodic termination."""

    source_model = SourceModel

    def _goal_states(self, source):
        return (source.size_of_one_room // 2,)
