"""Collect both pellets in the catalogue's unchanged single-ghost Pac-Man.

State s=((player*cells+ghost)<<2)|mask records two positions and two pellet
bits, plus win and lose terminals. Actions 0..3 move up, down, left and
right, clipped at boundaries. The ghost chooses uniformly among distinct
neighbor outcomes. Collision or crossing loses before complete collection
wins; both explicit terminals remain absorbing. Intermediate pellet bonuses
are removed, while all transitions and action availability remain unchanged.

state_dim is a positive Python integer target (default 326); width=max(3,
round((max(0,state_dim-2)/4)**0.25)), cells=width**2, S=4*cells**2+2, A=4.
Rewards are 1 for nonterminal actions with positive probability of entering
win, 0 elsewhere and at both terminals. This is an action reward, not a
success probability. Each finite instance has a positive minimum win-entry
probability among rewarded actions, giving finite expected total reward.
No time limit or seed is added; cycles prevent the acyclic certificate.
Reference: QVBS Pacman; this remains the local reduced model, not a
conversion of the published JANI source.
"""

from mdpforge.models.discounted.pacman import Model as DiscountedPacman

METADATA = {
    "category": "games_and_puzzles",
    "description": "Collect both pellets without capture, rewarding possible victory.",
    "reference": "QVBS Pacman benchmark: https://qcomp.org/benchmarks/",
    "sizes": {
        "small": {"requested_state_dim": 26400, "parameters": {}},
        "medium": {"requested_state_dim": 105600, "parameters": {}},
        "large": {"requested_state_dim": 422400, "parameters": {}},
    },
}

TEST_PARAMETERS = {"state_dim": 326}


class Model(DiscountedPacman):
    """Single-ghost Pac-Man with finite total reward for possible victory."""

    supported_criteria = frozenset({"total"})

    def __init__(self, state_dim: int = 326):
        """Accept state_dim as a positive Python integer target (default 326)."""
        super().__init__(state_dim=state_dim)
        self.name += "_total_win_entry_reward_v1"
        self.terminal_states = (self.win, self.lose)

    def _build_model(self):
        """Keep ghost and collection transitions, rewarding only possible wins."""
        super()._build_model()
        for action, transition in enumerate(self.transition_matrix):
            self.reward_matrix[:, action] = (
                transition[:, self.win].toarray().ravel() > 0
            )
        self.reward_matrix[list(self.terminal_states), :] = 0.0
