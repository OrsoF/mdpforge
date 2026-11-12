"""Finite-return sutton model with a one-percent per-step timeout."""

from mdpforge.models.discounted.sutton import Model as SourceModel
from mdpforge.models.total._episodic import EpisodicModel

METADATA = {
    "category": "navigation",
    "description": (
        "Reach the finish line on Sutton's racetrack with a 1% step timeout."
    ),
    "reference": (
        "Sutton, R. S. and Barto, A. G. (2018). Reinforcement Learning: An "
        "Introduction, 2nd ed., Exercise 5.12."
    ),
    "sizes": {
        "small": {"requested_state_dim": 6800, "parameters": {}},
        "medium": {"requested_state_dim": 105000, "parameters": {}},
        "large": {"requested_state_dim": 217600, "parameters": {}},
    },
}

TEST_PARAMETERS = {"state_dim": 140}


class Model(EpisodicModel):
    """Reach the finish line on Sutton's racetrack with episodic termination."""

    source_model = SourceModel

    def _goal_states(self, source):
        return (source.state_dim - 1,)
