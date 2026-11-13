"""Collect a treasure using the catalogue's unchanged underwater dynamics.

State s=(depth*3+column)*(oxygen_capacity+1)+oxygen records three columns
and oxygen, plus treasure and drowned terminals. Actions 0..4 descend,
ascend, move left, move right and collect/refill. At the surface action 4
refills oxygen; underwater it collects only in column 2 with oxygen left,
otherwise it stays in place unless oxygen is exhausted. Movement consumes
oxygen, with rightward currents and depth-dependent drowning risk. Both
terminals stay absorbing; all clipped and invalid moves are unchanged.

state_dim is a positive Python integer target (default 542); maximum depth
d=max(2, round((-3+sqrt(1+2*(state_dim-2)/3))/2)), oxygen_capacity=2*d+3,
S=3*(d+1)*(oxygen_capacity+1)+2, A=5. current_probability is in [0,1]
(default 0.15). No random generation, seed or time limit is added.

Rewards replace all costs and depth-dependent payoffs: 1 for nonterminal
actions entering treasure, 0 otherwise and at both terminals. All treasures
now have equal reward. Collection is deterministic, so total reward is at
most 1 even without termination; cycles prevent the acyclic certificate.
Reference: ElSayed-Aly, Parker & Feng, NFM 2024 DS Treasure; this retains
the local reduced model, not the original distributional formulation.
"""

from mdpforge.models.discounted.deep_sea_treasure import (
    Model as DiscountedDeepSeaTreasure,
)

METADATA = {
    "category": "navigation",
    "description": "Collect an underwater treasure while managing oxygen and currents.",
    "reference": (
        "ElSayed-Aly, Parker & Feng, NFM 2024, DS Treasure example: "
        "https://www.prismmodelchecker.org/files/nfm24dpmc/"
    ),
    "sizes": {
        "tiny": {"requested_state_dim": 2400, "parameters": {}},
        "small": {"requested_state_dim": 24000, "parameters": {}},
        "medium": {"requested_state_dim": 192000, "parameters": {}},
        "large": {"requested_state_dim": 768000, "parameters": {}},
    },
}

TEST_PARAMETERS = {"state_dim": 74}


class Model(DiscountedDeepSeaTreasure):
    """Underwater collection with finite total reward for treasure access."""

    supported_criteria = frozenset({"total"})

    def __init__(self, state_dim: int = 542, current_probability: float = 0.15):
        """Use positive integer state_dim and current_probability in [0,1]."""
        super().__init__(state_dim=state_dim, current_probability=current_probability)
        self.name = (
            f"{self.state_dim}_{self.action_dim}_treasure_total_1"
            f"_{self.depth}_{self.current_probability}"
        )
        self.terminal_states = (self.treasure, self.drowned)

    def _build_model(self):
        """Keep underwater transitions and reward only treasure collection."""
        super()._build_model()
        for action, transition in enumerate(self.transition_matrix):
            self.reward_matrix[:, action] = (
                transition[:, self.treasure].toarray().ravel() > 0
            )
        self.reward_matrix[list(self.terminal_states), :] = 0.0
