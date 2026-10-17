"""Reduced shared-coin consensus random walk, not Aspnes--Herlihy PRISM code."""

from mdpforge.core.mdp import MDP
from mdpforge.models._benchmark_sparse import TransitionBuilder

METADATA = {
    "category": "abstract",
    "description": (
        "Reduced shared coin: fair coin flips, bounded counter and two decisions."
    ),
    "reference": (
        "Aspnes and Herlihy (1990); PRISM consensus MDP: "
        "https://www.prismmodelchecker.org/benchmarks/models.php#mdps"
    ),
    "sizes": {
        "small": {"requested_state_dim": 2100, "parameters": {"processes": 3}},
        "medium": {"requested_state_dim": 506, "parameters": {"processes": 3}},
        "large": {"requested_state_dim": 4090, "parameters": {"processes": 3}},
    },
}

TEST_PARAMETERS = {"state_dim": 58, "processes": 3}


class Model(MDP):
    """State (shared_walk_position, local_coin_mask), excluding terminal bounds."""

    def __init__(self, state_dim: int = 122, processes: int = 3):
        if type(state_dim) is not int or state_dim < 1:
            raise ValueError("state_dim must be a positive integer")
        if type(processes) is not int or not 2 <= processes <= 10:
            raise ValueError("processes must be in [2, 10]")
        self.processes = processes
        self.masks = 1 << processes
        self.bound = max(2, round(((state_dim - 2) / self.masks + 1) / 2))
        super().__init__((2 * self.bound - 1) * self.masks + 2, processes)
        self.requested_state_dim = state_dim
        self.decide_zero, self.decide_one = self.state_dim - 2, self.state_dim - 1
        self.name = (
            f"{self.state_dim}_{self.action_dim}_consensus_v1"
            f"_n{processes}_bound{self.bound}"
        )

    def _encode(self, position, mask):
        return (position + self.bound - 1) * self.masks + mask

    def _build_model(self):
        builder = TransitionBuilder(self.state_dim, self.action_dim)
        for position in range(1 - self.bound, self.bound):
            for mask in range(self.masks):
                s = self._encode(position, mask)
                for a in range(self.processes):
                    builder.rewards[s, a] = -1
                    for flip, movement in ((0, -1), (1, 1)):
                        next_position = position + movement
                        next_mask = (mask & ~(1 << a)) | (flip << a)
                        if next_position == self.bound:
                            target = self.decide_one
                            builder.rewards[s, a] += 0.5 * (
                                20 if next_mask == self.masks - 1 else -5
                            )
                        elif next_position == -self.bound:
                            target = self.decide_zero
                            builder.rewards[s, a] += 0.5 * (
                                20 if next_mask == 0 else -5
                            )
                        else:
                            target = self._encode(next_position, next_mask)
                        builder.add(a, s, target, 0.5)
        builder.absorbing(self.decide_zero)
        builder.absorbing(self.decide_one)
        self.transition_matrix = builder.finish()
        self.reward_matrix = builder.rewards
