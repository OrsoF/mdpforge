"""Reduced IEEE 1394 FireWire leader election with digital countdowns.

States (timer_A, timer_B, retry_round, phase), two absorbing results.
At phase 0 the scheduler selects short/long or random contention timers;
phase 1 decrements them, and the earliest expiration wins. Simultaneous
expiration causes a retry and may exhaust the deadline. This is not the
original PRISM PTA-to-MDP conversion and contains no real-time semantics.
"""

from mdpforge.core.mdp import MDP
from mdpforge.models._benchmark_sparse import TransitionBuilder

METADATA = {
    "category": "abstract",
    "description": "FireWire root election with probabilistic ties, timers, and retry limit.",
    "reference": (
        "PRISM IEEE 1394 FireWire root contention (KNS03b): "
        "https://www.prismmodelchecker.org/benchmarks/models.php#mdps"
    ),
    "sizes": {
        "small": {"requested_state_dim": 740, "parameters": {}},
        "medium": {"requested_state_dim": 514, "parameters": {}},
        "large": {"requested_state_dim": 5002, "parameters": {}},
    },
}
TEST_PARAMETERS = {"state_dim": 74}


class Model(MDP):
    """Actions 0 favour node A, 1 favour B, 2 symmetric random contention."""

    def __init__(
        self,
        state_dim: int = 514,
        max_rounds: int = 3,
        tie_probability: float = 0.2,
    ):
        if type(state_dim) is not int or state_dim < 1:
            raise ValueError("state_dim must be a positive integer")
        if type(max_rounds) is not int or max_rounds < 1:
            raise ValueError("max_rounds must be a positive integer")
        if not 0 <= tie_probability <= 1:
            raise ValueError("tie_probability must be in [0, 1]")
        self.max_rounds = max_rounds
        self.tie_probability = float(tie_probability)
        self.delay = max(
            2, round(((state_dim - 2) / (2 * (max_rounds + 1))) ** 0.5) - 1
        )
        super().__init__(2 * (self.delay + 1) ** 2 * (max_rounds + 1) + 2, 3)
        self.requested_state_dim = state_dim
        self.elected, self.deadline = self.state_dim - 2, self.state_dim - 1
        self.name = (
            f"{self.state_dim}_{self.action_dim}_firewire_v1"
            f"_delay{self.delay}_rounds{max_rounds}_tie{tie_probability}"
        )

    def _encode(self, a, b, rounds, phase):
        return ((a * (self.delay + 1) + b) * (self.max_rounds + 1) + rounds) * 2 + phase

    def _build_model(self):
        t = TransitionBuilder(self.state_dim, self.action_dim)
        for a in range(self.delay + 1):
            for b in range(self.delay + 1):
                for rounds in range(self.max_rounds + 1):
                    for phase in (0, 1):
                        s = self._encode(a, b, rounds, phase)
                        for action in range(3):
                            t.rewards[s, action] = -1
                            if phase == 0:
                                # Equal timers contend and need another round.
                                if action == 0:
                                    times = (
                                        (1, self.delay, 1 - self.tie_probability),
                                        (1, 1, self.tie_probability),
                                    )
                                elif action == 1:
                                    times = (
                                        (self.delay, 1, 1 - self.tie_probability),
                                        (1, 1, self.tie_probability),
                                    )
                                else:
                                    times = (
                                        (1, self.delay, (1 - self.tie_probability) / 2),
                                        (self.delay, 1, (1 - self.tie_probability) / 2),
                                        (1, 1, self.tie_probability),
                                    )
                                for next_a, next_b, p in times:
                                    t.add(
                                        action,
                                        s,
                                        self._encode(next_a, next_b, rounds, 1),
                                        p,
                                    )
                            elif a == 0 and b == 0:
                                if rounds >= self.max_rounds:
                                    t.add(action, s, self.deadline, 1)
                                    t.rewards[s, action] -= 25
                                else:
                                    t.add(
                                        action, s, self._encode(0, 0, rounds + 1, 0), 1
                                    )
                                    t.rewards[s, action] -= 2
                            elif a == 0 or b == 0:
                                t.add(action, s, self.elected, 1)
                                t.rewards[s, action] += 30
                            else:
                                t.add(
                                    action, s, self._encode(a - 1, b - 1, rounds, 1), 1
                                )
        t.absorbing(self.elected)
        t.absorbing(self.deadline)
        self.transition_matrix, self.reward_matrix = t.finish(), t.rewards
