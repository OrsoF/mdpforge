"""Finite-return ninerooms model with a one-percent per-step timeout."""

from mdpforge.models.discounted.ninerooms import Model as SourceModel
from mdpforge.models.total._episodic import EpisodicModel

METADATA = {
    "category": "navigation",
    "description": "Reach the exit through nine connected rooms with a 1% step timeout.",
    "reference": "Inspired by Hengst, B. (2012). Hierarchical approaches.",
    "sizes": {
        "small": {"requested_state_dim": 18000, "parameters": {}},
        "medium": {"requested_state_dim": 72000, "parameters": {}},
        "large": {"requested_state_dim": 288000, "parameters": {}},
    },
}

TEST_PARAMETERS = {"state_dim": 81}


class Model(EpisodicModel):
    """Reach the exit through nine connected rooms with episodic termination."""

    source_model = SourceModel

    def _goal_states(self, source):
        return (source.side_size // 6,)
