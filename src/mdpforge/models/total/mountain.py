"""Finite-return mountain model with a one-percent per-step timeout."""

import numpy as np

from mdpforge.models.discounted.mountain import Model as SourceModel
from mdpforge.models.total._episodic import EpisodicModel

METADATA = {
    "category": "physical_control",
    "description": "Reach the Mountain Car goal with a 1% step timeout.",
    "reference": (
        "Moore, A. W. (1990). Efficient Memory-based Learning for Robot Control. "
        "PhD thesis, University of Cambridge."
    ),
    "sizes": {
        "small": {"requested_state_dim": 10800, "parameters": {}},
        "medium": {"requested_state_dim": 43200, "parameters": {}},
        "large": {"requested_state_dim": 172800, "parameters": {}},
    },
}

TEST_PARAMETERS = {"state_dim": 100}


class Model(EpisodicModel):
    """Reach the Mountain Car goal with episodic termination."""

    source_model = SourceModel

    def _goal_states(self, source):
        positions = np.linspace(-1.2, 0.6, source.x_axis_states_number)
        return tuple(
            position * source.speed_states_number + speed
            for position, x in enumerate(positions)
            if x > 0.5
            for speed in range(source.speed_states_number)
        )
