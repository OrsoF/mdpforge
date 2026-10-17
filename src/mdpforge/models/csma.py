"""Reduced two-station CSMA/CD with bounded randomized binary backoff."""

from mdpforge.core.mdp import MDP
from mdpforge.models._benchmark_sparse import TransitionBuilder

METADATA = {
    "category": "service_operations",
    "description": (
        "Reduced CSMA/CD: two stations, contention, random backoff and collision limit."
    ),
    "reference": (
        "PRISM IEEE 802.3 CSMA/CD benchmark (KNSW07): "
        "https://www.prismmodelchecker.org/benchmarks/models.php#mdps"
    ),
    "sizes": {
        "small": {"requested_state_dim": 110, "parameters": {"backoff": 2}},
        "medium": {"requested_state_dim": 578, "parameters": {"backoff": 3}},
        "large": {"requested_state_dim": 3102, "parameters": {"backoff": 4}},
    },
}

TEST_PARAMETERS = {"state_dim": 110, "backoff": 2}


class Model(MDP):
    """State (delivered_bitmask, collisions, backoff0, backoff1) plus terminals."""

    def __init__(self, state_dim: int = 110, backoff: int = 2):
        if type(state_dim) is not int or state_dim < 1:
            raise ValueError("state_dim must be a positive integer")
        if type(backoff) is not int or not 1 <= backoff <= 10:
            raise ValueError("backoff must be in [1, 10]")
        self.backoff = backoff
        self.collisions = max(1, round((state_dim - 2) / (4 * (backoff + 1) ** 2)) - 1)
        super().__init__(4 * (self.collisions + 1) * (backoff + 1) ** 2 + 2, 3)
        self.requested_state_dim = state_dim
        self.success, self.failure = self.state_dim - 2, self.state_dim - 1
        self.name = (
            f"{self.state_dim}_{self.action_dim}_csma_v1"
            f"_window{backoff}_collisions{self.collisions}"
        )

    def _encode(self, mask, collisions, b0, b1):
        return (
            (mask * (self.collisions + 1) + collisions) * (self.backoff + 1) + b0
        ) * (self.backoff + 1) + b1

    def _build_model(self):
        builder = TransitionBuilder(self.state_dim, self.action_dim)
        for mask in range(4):
            for collisions in range(self.collisions + 1):
                for b0 in range(self.backoff + 1):
                    for b1 in range(self.backoff + 1):
                        s = self._encode(mask, collisions, b0, b1)
                        for action in range(3):
                            if mask == 3:
                                builder.add(action, s, self.success, 1)
                                continue
                            # Non-transmitting stations still count down their timers.
                            counters = (max(0, b0 - 1), max(0, b1 - 1))
                            requested = (1, 2, 3)[action]
                            sending = sum(
                                1 << i
                                for i, b in enumerate((b0, b1))
                                if (requested & (1 << i))
                                and not (mask & (1 << i))
                                and b == 0
                            )
                            builder.rewards[s, action] = -1
                            if sending == 3:
                                builder.rewards[s, action] = -3
                                if collisions == self.collisions:
                                    builder.add(action, s, self.failure, 1)
                                else:
                                    window = min(
                                        self.backoff + 1, 2 ** (collisions + 1)
                                    )
                                    p = 1 / (window * window)
                                    for t0 in range(window):
                                        for t1 in range(window):
                                            builder.add(
                                                action,
                                                s,
                                                self._encode(
                                                    mask, collisions + 1, t0, t1
                                                ),
                                                p,
                                            )
                            else:
                                next_mask = mask | sending
                                builder.rewards[s, action] += 8 * sending.bit_count()
                                target = (
                                    self.success
                                    if next_mask == 3
                                    else self._encode(next_mask, collisions, *counters)
                                )
                                builder.add(action, s, target, 1)
        builder.absorbing(self.success)
        builder.absorbing(self.failure)
        self.transition_matrix = builder.finish()
        self.reward_matrix = builder.rewards
