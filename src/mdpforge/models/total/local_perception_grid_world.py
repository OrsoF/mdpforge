"""Reach the local-perception maze's goal while retaining true cell states.

State s=row*width+column is the true cell, not its aliased observation.
Actions 0..3 are north, south, east and west; blocked moves stay in place.
Walls retain the discounted geometry. The goal at (1, width-2) remains
absorbing. Adjacent-wall bits in north/south/east/west order define 16
observations, preserved in observation_map and observation_matrix.

state_dim is a positive Python integer target (default 49). With target=
max(49, state_dim), height=max(7, floor(sqrt(target))) and width=max(7,
ceil(target/height)), actual S=height*width and A=4. No seed is used.
Transitions are unchanged; rewards are 1 for nonterminal goal-entry
actions and 0 otherwise, including at the goal. Total reward is at most
1 even for nonterminating policies. No time limit is added; cycles prevent
the acyclic certificate. Observation data are rebuilt after cache loads.
Reference: McCallum (1996), Selective Perception and Hidden State;
this retains the catalogue's concrete-state maze abstraction.
"""

import numpy as np

from mdpforge.models.discounted.local_perception_grid_world import (
    Model as DiscountedLocalPerceptionGridWorld,
)

METADATA = {
    "category": "navigation",
    "description": "Reach a maze goal while retaining aliased local-wall observations.",
    "reference": (
        "McCallum (1996), Reinforcement Learning with Selective Perception and "
        "Hidden State."
    ),
    "sizes": {
        "small": {"requested_state_dim": 11600, "parameters": {}},
        "medium": {"requested_state_dim": 46400, "parameters": {}},
        "large": {"requested_state_dim": 185600, "parameters": {}},
    },
}

TEST_PARAMETERS = {"state_dim": 49}


class Model(DiscountedLocalPerceptionGridWorld):
    """Concrete-state maze with separate observations and finite total reward."""

    supported_criteria = frozenset({"total"})

    def __init__(self, state_dim: int = 49):
        """Accept state_dim as a positive Python integer target (default 49)."""
        if type(state_dim) is not int or state_dim <= 0:
            raise ValueError("state_dim must be a positive integer")
        super().__init__(state_dim=state_dim)
        self.name += "_total_goal_entry_reward_v1"
        self.goal_cell = self._cell(1, self.width - 2)
        self.terminal_states = (self.goal_cell,)

    def _build_model(self):
        """Keep maze transitions and observations, replacing only rewards."""
        super()._build_model()
        for action, transition in enumerate(self.transition_matrix):
            self.reward_matrix[:, action] = (
                transition[:, self.goal_cell].toarray().ravel() > 0
            )
        self.reward_matrix[self.goal_cell, :] = 0.0

    def create_model(self, check_transition: bool = False, *, cache: str = "readwrite"):
        """Use inherited construction and cache options, restoring missing observations."""
        super().create_model(check_transition=check_transition, cache=cache)
        if not hasattr(self, "observation_map"):
            self.blocked = self._make_walls()
            self.observation_map = np.fromiter(
                (self._observe(state) for state in range(self.state_dim)),
                dtype=int,
                count=self.state_dim,
            )
            self.observation_matrix = np.zeros((self.state_dim, self.observation_dim))
            self.observation_matrix[
                np.arange(self.state_dim), self.observation_map
            ] = 1.0
