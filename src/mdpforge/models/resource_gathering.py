"""Reduced QVBS-inspired gathering MDP with stochastic traversal and enemies."""

import math

from mdpforge.core.mdp import MDP
from mdpforge.models._benchmark_sparse import TransitionBuilder

METADATA = {
    "category": "asset_management",
    "description": (
        "Reduced resource gathering: collect gold/gem, manage energy, "
        "evade hazard and return home."
    ),
    "reference": "QVBS resource-gathering benchmark: https://qcomp.org/benchmarks/",
    "sizes": {
        "small": {"requested_state_dim": 514, "parameters": {"energy": 7}},
        "medium": {"requested_state_dim": 1730, "parameters": {"energy": 11}},
        "large": {"requested_state_dim": 5834, "parameters": {"energy": 17}},
    },
}

TEST_PARAMETERS = {"state_dim": 514, "energy": 7}


class Model(MDP):
    """State (position, resources bitmask, energy), plus banked and failed."""

    def __init__(
        self, state_dim: int = 514, energy: int = 7,
        slip_probability: float = 0.1, hazard_probability: float = 0.2,
    ):
        if type(state_dim) is not int or state_dim < 1:
            raise ValueError("state_dim must be a positive integer")
        if type(energy) is not int or not 1 <= energy <= 30:
            raise ValueError("energy must be an integer in [1, 30]")
        if not 0 <= slip_probability <= 1 or not 0 <= hazard_probability <= 1:
            raise ValueError("probabilities must be in [0, 1]")
        self.energy = energy
        self.slip_probability = float(slip_probability)
        self.hazard_probability = float(hazard_probability)
        self.width = max(3, round(math.sqrt((state_dim - 2) / (4 * (energy + 1)))))
        self.cells = self.width**2
        super().__init__(self.cells * 4 * (energy + 1) + 2, 5)
        self.requested_state_dim = state_dim
        self.banked, self.failed = self.state_dim - 2, self.state_dim - 1
        self.gold = self.width - 1
        self.gem = self.cells - 1
        self.hazard = (self.width // 2) * self.width + self.width // 2
        self.name = (
            f"resource_gathering_v1_grid{self.width}_energy{energy}"
            f"_slip{slip_probability:.17g}_hazard{hazard_probability:.17g}"
        )

    def _encode(self, position, mask, energy):
        return (position * 4 + mask) * (self.energy + 1) + energy

    def _step(self, position, action):
        y, x = divmod(position, self.width)
        dy, dx = ((-1, 0), (1, 0), (0, -1), (0, 1))[action]
        return min(self.width - 1, max(0, y + dy)) * self.width + min(
            self.width - 1, max(0, x + dx)
        )

    def _build_model(self):
        builder = TransitionBuilder(self.state_dim, self.action_dim)
        for position in range(self.cells):
            for mask in range(4):
                for energy in range(self.energy + 1):
                    s = self._encode(position, mask, energy)
                    for action in range(5):
                        builder.rewards[s, action] = -1
                        if action == 4:
                            if position == 0 and mask:
                                builder.add(action, s, self.banked, 1)
                                builder.rewards[s, action] = (
                                    10 * bool(mask & 1) + 20 * bool(mask & 2)
                                )
                            elif position == 0:
                                builder.add(
                                    action,
                                    s,
                                    self._encode(position, mask, self.energy),
                                    1,
                                )
                                builder.rewards[s, action] = -2
                            elif position in (self.gold, self.gem):
                                new_mask = mask | (1 if position == self.gold else 2)
                                # Resource sites also serve as recharge depots.
                                builder.add(
                                    action,
                                    s,
                                    self._encode(position, new_mask, self.energy),
                                    1,
                                )
                                if new_mask == mask:
                                    builder.rewards[s, action] = -2
                            else:
                                builder.add(action, s, s, 1)
                                builder.rewards[s, action] = -2
                        elif not energy:
                            builder.add(action, s, self.failed, 1)
                            builder.rewards[s, action] = -20
                        else:
                            for actual in range(4):
                                p = (
                                    1 - self.slip_probability
                                    if action == actual
                                    else self.slip_probability / 3
                                )
                                dest = self._step(position, actual)
                                if dest == self.hazard:
                                    builder.add(
                                        action,
                                        s,
                                        self.failed,
                                        p * self.hazard_probability,
                                    )
                                    builder.rewards[s, action] -= (
                                        p * self.hazard_probability * 20
                                    )
                                    p *= 1 - self.hazard_probability
                                builder.add(
                                    action, s, self._encode(dest, mask, energy - 1), p
                                )
        builder.absorbing(self.banked)
        builder.absorbing(self.failed)
        self.transition_matrix = builder.finish()
        self.reward_matrix = builder.rewards
