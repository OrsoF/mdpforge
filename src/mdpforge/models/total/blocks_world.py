"""Complete the final Blocks World puzzle with unchanged interlevel dynamics.

Each 96-state level encodes (focus, red_cover, blue_cover, holding, done)
in mixed radix (3,2,2,4,2). Actions 0..5 look at red, blue or green,
pick, put aside and lift green. Invalid actions retain the original state
changes. Lifting exposed green on a higher level enters the next lower
level's start; level 0 ends at a done state. All existing done states
remain absorbing, including higher-level states unreachable by completion.
Only level-0 done states count as success goals.

state_dim is a positive Python integer (default 96), giving exact S and
A=6. divmod(S,96) gives the number of levels and preparation states. The
preparation chain uses action 3 to enter the first puzzle; for S<96 it
retains the existing chain-only task with state 0 as its absorbing goal.
No states, transitions, time limits or seeds are added. Rewards replace
all intermediate bonuses and costs: 1 only for nonterminal actions entering
a success goal, 0 otherwise and at all terminals. Deterministic goal entry
bounds total reward by 1 even without termination. Self-loops prevent the
acyclic certificate. Reference: McCallum (1996), Whitehead's Blocks World
revisited; the catalogue scales by sequential levels, not richer puzzles.
"""

import numpy as np

from mdpforge.models.discounted.blocks_world import Model as DiscountedBlocksWorld

METADATA = {
    "category": "games_and_puzzles",
    "description": "Complete sequential block puzzles, rewarding only final completion.",
    "reference": "McCallum (1996), Whitehead's Blocks World revisited.",
    "sizes": {
        "small": {"requested_state_dim": 8000, "parameters": {}},
        "medium": {"requested_state_dim": 64000, "parameters": {}},
        "large": {"requested_state_dim": 256000, "parameters": {}},
    },
}

TEST_PARAMETERS = {"state_dim": 200}


class Model(DiscountedBlocksWorld):
    """Sequential block manipulation with finite total reward for final success."""

    supported_criteria = frozenset({"total"})

    def __init__(self, state_dim: int = 96):
        """Accept state_dim as a positive Python integer count (default 96)."""
        if type(state_dim) is not int or state_dim <= 0:
            raise ValueError("state_dim must be a positive integer")
        super().__init__(state_dim=state_dim)
        self.name += "_total_final_level_entry_reward_v1"
        if self.n_levels:
            self.terminal_states = tuple(range(1, self._world_limit, 2))
            self.goal_states = tuple(range(1, self._LEVEL_SIZE, 2))
        else:
            self.terminal_states = (0,)
            self.goal_states = (0,)

    def _build_model(self):
        """Keep every level transition and reward only final-goal entry."""
        super()._build_model()
        for action, transition in enumerate(self.transition_matrix):
            self.reward_matrix[:, action] = (
                np.asarray(transition[:, self.goal_states].sum(axis=1)).ravel() > 0
            )
        self.reward_matrix[list(self.terminal_states), :] = 0.0

    def transition(self, s: int, action: int) -> tuple[int, float]:
        """Return the unchanged successor and the total-reward action bonus."""
        next_state, _ = super().transition(s, action)
        if self.n_levels:
            terminal = s < self._world_limit and s % 2 == 1
        else:
            terminal = s == 0
        reward = float(not terminal and next_state in self.goal_states)
        return next_state, reward
