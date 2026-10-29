"""Reduced energy-constrained Deep Sea Treasure MDP (PRISM NFM'24 inspiration).

States (depth, column in {0,1,2}, oxygen), plus collected/dead terminals.
Actions: descend, ascend, left, right, collect/refill-at-surface.
A scalar discounted reward is used instead of the original distributional
or multiobjective properties; there is no claim of numerical equivalence.
"""

from mdpforge.core.mdp import MDP
from mdpforge.models._benchmark_sparse import TransitionBuilder

METADATA = {
    "category": "navigation",
    "description": "Depth-dependent treasure, limited oxygen, risky currents, and surfacing.",
    "reference": (
        "ElSayed-Aly, Parker & Feng, NFM 2024, DS Treasure example: "
        "https://www.prismmodelchecker.org/files/nfm24dpmc/"
    ),
    "sizes": {
        "small": {"requested_state_dim": 24000, "parameters": {}},
        "medium": {"requested_state_dim": 192000, "parameters": {}},
        "large": {"requested_state_dim": 768000, "parameters": {}},
    },
}
TEST_PARAMETERS = {"state_dim": 74}


class Model(MDP):
    """Size derives quadratic depth by matching 3*(depth+1)*(2*depth+4)+2."""

    def __init__(self, state_dim: int = 542, current_probability: float = 0.15):
        if type(state_dim) is not int or state_dim < 1:
            raise ValueError("state_dim must be a positive integer")
        if not 0 <= current_probability <= 1:
            raise ValueError("current_probability must be in [0, 1]")
        self.depth = max(2, round((-3 + (1 + 4 * (state_dim - 2) / 6) ** 0.5) / 2))
        self.oxygen_capacity = 2 * self.depth + 3
        self.current_probability = float(current_probability)
        super().__init__(3 * (self.depth + 1) * (self.oxygen_capacity + 1) + 2, 5)
        self.requested_state_dim = state_dim
        self.treasure, self.drowned = self.state_dim - 2, self.state_dim - 1
        self.name = (
            f"{self.state_dim}_{self.action_dim}_deep_sea_treasure_1"
            f"_{self.depth}_{self.current_probability}"
        )

    def _encode(self, depth, column, oxygen):
        return (depth * 3 + column) * (self.oxygen_capacity + 1) + oxygen

    def _build_model(self):
        t = TransitionBuilder(self.state_dim, self.action_dim)
        for depth in range(self.depth + 1):
            for column in range(3):
                for oxygen in range(self.oxygen_capacity + 1):
                    s = self._encode(depth, column, oxygen)
                    for action in range(5):
                        if action == 4 and depth == 0:
                            t.add(
                                action,
                                s,
                                self._encode(0, column, self.oxygen_capacity),
                                1,
                            )
                            t.rewards[s, action] = -0.3
                        elif oxygen == 0:
                            t.add(action, s, self.drowned, 1)
                            t.rewards[s, action] = -20
                        elif action == 4 and column == 2 and depth > 0:
                            t.add(action, s, self.treasure, 1)
                            t.rewards[s, action] = 2.5 * depth**1.5
                        elif action == 4:
                            t.add(action, s, s, 1)
                            t.rewards[s, action] = -2
                        else:
                            next_depth = max(
                                0,
                                min(self.depth, depth + (action == 0) - (action == 1)),
                            )
                            next_column = max(
                                0, min(2, column + (action == 3) - (action == 2))
                            )
                            # Side current pushes one column right, potentially saturating the edge.
                            t.rewards[s, action] = -0.35
                            hazard = 0.04 * (next_depth / self.depth)
                            for c, p_current in (
                                (next_column, 1 - self.current_probability),
                                (min(2, next_column + 1), self.current_probability),
                            ):
                                t.add(action, s, self.drowned, hazard * p_current)
                                t.add(
                                    action,
                                    s,
                                    self._encode(next_depth, c, oxygen - 1),
                                    (1 - hazard) * p_current,
                                )
                            t.rewards[s, action] -= 20 * hazard
        t.absorbing(self.treasure)
        t.absorbing(self.drowned)
        self.transition_matrix, self.reward_matrix = t.finish(), t.rewards
