"""Reduced IEEE 802.11 contention MDP, loosely inspired by PRISM WLAN PTA.

States (backoff_A, backoff_B, collision_count, delivered_mask) plus two
terminals. Actions choose symmetric, A-priority or B-priority access.
Counters tick when nonzero; ready stations transmit with policy-dependent
probabilities. Collision resets both stations' backoffs (bounded).
Original digital-clock semantics, timings and properties are not preserved.
"""

from mdpforge.core.mdp import MDP
from mdpforge.models._benchmark_sparse import TransitionBuilder

METADATA = {
    "category": "service_operations",
    "description": "Two-station Wi-Fi contention with backoffs, priorities and collisions.",
    "reference": (
        "PRISM IEEE 802.11 WLAN (KNS02a): "
        "https://www.prismmodelchecker.org/benchmarks/models.php#mdps"
    ),
    "sizes": {
        "small": {"requested_state_dim": 1050, "parameters": {}},
        "medium": {"requested_state_dim": 578, "parameters": {}},
        "large": {"requested_state_dim": 5186, "parameters": {}},
    },
}
TEST_PARAMETERS = {"state_dim": 66}


class Model(MDP):
    """Delivered-mask bits: A=1, B=2. Collisions capped at max_collisions."""

    def __init__(
        self,
        state_dim: int = 578,
        max_collisions: int = 3,
        external_interference: float = 0.05,
    ):
        if type(state_dim) is not int or state_dim < 1:
            raise ValueError("state_dim must be a positive integer")
        if type(max_collisions) is not int or max_collisions < 1:
            raise ValueError("max_collisions must be a positive integer")
        if not 0 <= external_interference <= 1:
            raise ValueError("external_interference must be in [0, 1]")
        self.max_collisions = max_collisions
        self.external_interference = float(external_interference)
        self.backoff_limit = max(
            1, round(((state_dim - 2) / (4 * (max_collisions + 1))) ** 0.5) - 1
        )
        super().__init__(
            4 * (self.backoff_limit + 1) ** 2 * (max_collisions + 1) + 2, 3
        )
        self.requested_state_dim = state_dim
        self.success, self.failure = self.state_dim - 2, self.state_dim - 1
        self.name = (
            f"{self.state_dim}_{self.action_dim}_wlan_v1"
            f"_b{self.backoff_limit}_col{max_collisions}"
            f"_interference{external_interference}"
        )

    def _encode(self, backoff_a, backoff_b, collisions, delivered):
        return (
            (backoff_a * (self.backoff_limit + 1) + backoff_b)
            * (self.max_collisions + 1)
            + collisions
        ) * 4 + delivered

    def _build_model(self):
        t = TransitionBuilder(self.state_dim, self.action_dim)
        for a in range(self.backoff_limit + 1):
            for b in range(self.backoff_limit + 1):
                for collisions in range(self.max_collisions + 1):
                    for delivered in range(4):
                        s = self._encode(a, b, collisions, delivered)
                        for action in range(3):
                            if delivered == 3:
                                t.add(action, s, self.success, 1)
                                t.rewards[s, action] = 24
                                continue
                            p_a, p_b = ((0.65, 0.65), (0.95, 0.25), (0.25, 0.95))[
                                action
                            ]
                            p_a = p_a if a == 0 and not delivered & 1 else 0
                            p_b = p_b if b == 0 and not delivered & 2 else 0
                            t.rewards[s, action] = -1
                            for send_a, pa in ((0, 1 - p_a), (1, p_a)):
                                for send_b, pb in ((0, 1 - p_b), (1, p_b)):
                                    p = pa * pb
                                    if not p:
                                        continue
                                    count = send_a + send_b
                                    # Interference leads to collision even with one station.
                                    for collision, pc in (
                                        (
                                            True,
                                            1
                                            if count == 2
                                            else (
                                                self.external_interference
                                                if count == 1
                                                else 0
                                            ),
                                        ),
                                        (
                                            False,
                                            0
                                            if count == 2
                                            else (
                                                1 - self.external_interference
                                                if count == 1
                                                else 1
                                            ),
                                        ),
                                    ):
                                        q = p * pc
                                        if not q:
                                            continue
                                        if collision:
                                            next_collisions = collisions + 1
                                            if next_collisions > self.max_collisions:
                                                t.add(action, s, self.failure, q)
                                                t.rewards[s, action] -= 25 * q
                                            else:
                                                backoff = min(
                                                    self.backoff_limit,
                                                    1 + next_collisions,
                                                )
                                                t.add(
                                                    action,
                                                    s,
                                                    self._encode(
                                                        backoff,
                                                        backoff,
                                                        next_collisions,
                                                        delivered,
                                                    ),
                                                    q,
                                                )
                                                t.rewards[s, action] -= 2 * q
                                        else:
                                            done = delivered | send_a | (send_b << 1)
                                            # Ready stations not transmitting remain ready.
                                            next_a = 0 if send_a else max(0, a - 1)
                                            next_b = 0 if send_b else max(0, b - 1)
                                            if done == 3:
                                                t.add(action, s, self.success, q)
                                                t.rewards[s, action] += 24 * q
                                            else:
                                                t.add(
                                                    action,
                                                    s,
                                                    self._encode(
                                                        next_a, next_b, collisions, done
                                                    ),
                                                    q,
                                                )
                                                t.rewards[s, action] += (
                                                    4 * (send_a + send_b) * q
                                                )
        t.absorbing(self.success)
        t.absorbing(self.failure)
        self.transition_matrix, self.reward_matrix = t.finish(), t.rewards
