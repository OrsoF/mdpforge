"""Reduced single-cargo Boxworld, inspired by IPPC 2004 PPDDL Boxworld.

State (truck position, crate position, on-truck bit), plus goal/damage.
Actions: drive next, drive previous, express next-two, pickup, drop.
One truck, one crate, linear road; not a grounded IPPC instance.
"""

from mdpforge.core.mdp import MDP
from mdpforge.utils._benchmark_sparse import TransitionBuilder

METADATA = {
    "category": "service_operations",
    "description": "One-crate transport with probabilistic delays and possible cargo damage.",
    "reference": (
        "Younes et al., JAIR 24 (2005), IPPC 2004 Boxworld: "
        "https://www.cs.cmu.edu/afs/cs/project/jair/pub/volume24/younes05a-html/"
    ),
    "sizes": {
        "small": {"requested_state_dim": 20000, "parameters": {}},
        "medium": {"requested_state_dim": 80000, "parameters": {}},
        "large": {"requested_state_dim": 320000, "parameters": {}},
    },
}
TEST_PARAMETERS = {"state_dim": 52}


class Model(MDP):
    """Crate starts at road location zero, destination at the far end."""

    def __init__(
        self,
        state_dim: int = 514,
        traffic_delay: float = 0.2,
        cargo_damage: float = 0.025,
    ):
        if type(state_dim) is not int or state_dim < 1:
            raise ValueError("state_dim must be a positive integer")
        if not 0 <= traffic_delay <= 1 or not 0 <= cargo_damage <= 1:
            raise ValueError("probabilities must be in [0, 1]")
        self.locations = max(3, round((max(0, state_dim - 2) / 2) ** 0.5))
        self.traffic_delay = float(traffic_delay)
        self.cargo_damage = float(cargo_damage)
        super().__init__(2 * self.locations**2 + 2, 5)
        self.requested_state_dim = state_dim
        self.delivered, self.damaged = self.state_dim - 2, self.state_dim - 1
        self.name = (
            f"{self.state_dim}_{self.action_dim}_boxworld_1"
            f"_{self.locations}_{traffic_delay}_{cargo_damage}"
        )

    def _encode(self, truck, crate, loaded):
        return (truck * self.locations + crate) * 2 + loaded

    def _build_model(self):
        t = TransitionBuilder(self.state_dim, self.action_dim)
        for truck in range(self.locations):
            for crate in range(self.locations):
                for loaded in (0, 1):
                    s = self._encode(truck, crate, loaded)
                    for action in range(5):
                        t.rewards[s, action] = -1
                        if action == 3 and not loaded and truck == crate:
                            t.add(action, s, self._encode(truck, crate, 1), 1)
                        elif action == 4 and loaded:
                            if truck == self.locations - 1:
                                t.add(action, s, self.delivered, 1)
                                t.rewards[s, action] += 65
                            else:
                                t.add(action, s, self._encode(truck, truck, 0), 1)
                        elif action in (0, 1, 2):
                            target = max(
                                0,
                                min(
                                    self.locations - 1,
                                    truck
                                    + (
                                        -1 if action == 1 else (2 if action == 2 else 1)
                                    ),
                                ),
                            )
                            p_delay = min(
                                1.0, self.traffic_delay + 0.15 * (action == 2)
                            )
                            p_damage = (
                                self.cargo_damage * (2 if action == 2 else 1)
                                if loaded
                                else 0
                            )
                            t.add(action, s, self.damaged, p_damage)
                            t.add(
                                action,
                                s,
                                self._encode(target, crate, loaded),
                                (1 - p_damage) * (1 - p_delay),
                            )
                            t.add(
                                action,
                                s,
                                self._encode(truck, crate, loaded),
                                (1 - p_damage) * p_delay,
                            )
                            t.rewards[s, action] -= (
                                1 if action == 2 else 0
                            ) + 35 * p_damage
                        else:
                            t.add(action, s, s, 1)
                            t.rewards[s, action] = -3
        t.absorbing(self.delivered)
        t.absorbing(self.damaged)
        self.transition_matrix, self.reward_matrix = t.finish(), t.rewards
