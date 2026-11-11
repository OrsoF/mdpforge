"""Finite-return barto model with a one-percent per-step timeout."""

from mdpforge.models.discounted.barto import Model as SourceModel
from mdpforge.models.total._episodic import EpisodicModel

METADATA = {
    "category": "navigation",
    "description": "Reach the finish line on Barto's racetrack with a 1% step timeout.",
    "reference": (
        "Barto, A. G., Bradtke, S. J. and Singh, S. P. (1995). Learning to act "
        "using real-time dynamic programming. Artificial Intelligence 72(1-2):81-138."
    ),
    "sizes": {
        "small": {"requested_state_dim": 9000, "parameters": {}},
        "medium": {"requested_state_dim": 90000, "parameters": {}},
        "large": {"requested_state_dim": 288000, "parameters": {}},
    },
}

TEST_PARAMETERS = {"state_dim": 300}


class Model(EpisodicModel):
    """Reach the finish line on Barto's racetrack with episodic termination."""

    source_model = SourceModel

    def _goal_states(self, source):
        return tuple(
            i
            for i, (x, y, _, _) in enumerate(source.state_space)
            if source.is_finish_cell(x, y)
        )
