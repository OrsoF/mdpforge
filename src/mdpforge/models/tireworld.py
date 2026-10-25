"""Reduced Tireworld route problem with renewable spare-tire depots."""

from mdpforge.core.mdp import MDP
from mdpforge.models._benchmark_sparse import TransitionBuilder

METADATA = {
    "category": "navigation",
    "description": (
        "Reduced Tireworld: two-lane road, punctures, spare depots and dead ends."
    ),
    "reference": (
        "IPPC 2004 Tireworld (Younes et al., JAIR 2005): "
        "https://ipc04.icaps-conference.org/probabilistic/"
    ),
    "sizes": {
        "small": {"requested_state_dim": 15600, "parameters": {}},
        "medium": {"requested_state_dim": 62400, "parameters": {}},
        "large": {"requested_state_dim": 249600, "parameters": {}},
    },
}

TEST_PARAMETERS = {"state_dim": 50}


class Model(MDP):
    """State (column, lane, spare, flat), plus goal and stranded terminals."""

    # Action 0: forward in fast lane; 1: forward in depot lane;
    # 2: change lane without forward movement; 3: load; 4: repair.
    def __init__(self, state_dim: int = 98, puncture_probability: float = 0.15):
        if type(state_dim) is not int or state_dim < 1:
            raise ValueError("state_dim must be a positive integer")
        if not 0 <= puncture_probability <= 1:
            raise ValueError("puncture_probability must be in [0, 1]")
        self.length = max(3, round((state_dim - 2) / 8))
        self.puncture_probability = float(puncture_probability)
        super().__init__(8 * self.length + 2, 5)
        self.requested_state_dim = state_dim
        self.goal, self.stranded = self.state_dim - 2, self.state_dim - 1
        self.name = (
            f"{self.state_dim}_{self.action_dim}_tireworld_1"
            f"_{self.length}_{self.puncture_probability}"
        )

    def _encode(self, column, lane, spare, flat):
        return 4 * (2 * column + lane) + 2 * spare + flat

    def _build_model(self):
        builder = TransitionBuilder(self.state_dim, self.action_dim)
        for column in range(self.length):
            for lane in (0, 1):
                for spare in (0, 1):
                    for flat in (0, 1):
                        s = self._encode(column, lane, spare, flat)
                        for action in range(5):
                            builder.rewards[s, action] = -1
                            if column == self.length - 1:
                                builder.add(action, s, self.goal, 1)
                                builder.rewards[s, action] = 0
                            elif action == 3 and lane == 1 and not spare:
                                builder.add(
                                    action, s, self._encode(column, lane, 1, flat), 1
                                )
                            elif action == 4 and flat and spare:
                                builder.add(
                                    action, s, self._encode(column, lane, 0, 0), 1
                                )
                            elif action in (0, 1, 2) and not flat:
                                target_column = column + (action in (0, 1))
                                target_lane = 1 - lane if action == 2 else action
                                # Each driven edge has the same puncture rate.
                                p = self.puncture_probability
                                safe = self._encode(
                                    target_column, target_lane, spare, 0
                                )
                                broken = self._encode(
                                    target_column, target_lane, spare, 1
                                )
                                if target_column == self.length - 1:
                                    builder.add(action, s, self.goal, 1)
                                    builder.rewards[s, action] += 20
                                else:
                                    builder.add(action, s, safe, 1 - p)
                                    if not spare and target_lane == 0:
                                        builder.add(action, s, self.stranded, p)
                                        builder.rewards[s, action] -= 30 * p
                                    else:
                                        builder.add(action, s, broken, p)
                            elif flat and not spare and lane == 0:
                                builder.add(action, s, self.stranded, 1)
                                builder.rewards[s, action] = -30
                            else:
                                builder.add(action, s, s, 1)
                                builder.rewards[s, action] = -2
        builder.absorbing(self.goal)
        builder.absorbing(self.stranded)
        self.transition_matrix = builder.finish()
        self.reward_matrix = builder.rewards
