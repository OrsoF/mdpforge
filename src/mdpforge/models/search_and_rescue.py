"""Finite helicopter Search-and-Rescue MDP, inspired by IPPC 2008.

State (zone, local landing-site knowledge, survivor onboard, time budget).
Flight to another zone forgets previous landing observations: this is a
reduced, Markovian surrogate, *not* a faithful PPDDL ground model.
Actions: fly next, fly previous, survey site, pickup, evacuate/return.
"""

from mdpforge.core.mdp import MDP
from mdpforge.models._benchmark_sparse import TransitionBuilder

METADATA = {
    "category": "navigation",
    "description": "Helicopter survey, safe landing discovery, extraction and time pressure.",
    "reference": (
        "IPPC 2008 Search-and-Rescue (Teichteil-Konigsbuch): "
        "https://ipc08.icaps-conference.org/probabilistic/wiki/images/0/03/Results.pdf"
    ),
    "sizes": {
        "small": {"requested_state_dim": 22800, "parameters": {}},
        "medium": {"requested_state_dim": 91200, "parameters": {}},
        "large": {"requested_state_dim": 364800, "parameters": {}},
    },
}
TEST_PARAMETERS = {"state_dim": 74}


class Model(MDP):
    """Site knowledge 0=unknown, 1=safe, 2=unsafe. Zone 0 is home base."""

    def __init__(
        self,
        state_dim: int = 506,
        max_urgency: int = 3,
        survey_success: float = 0.65,
        pickup_success: float = 0.85,
        per_step_loss: float = 0.05,
    ):
        if type(state_dim) is not int or state_dim < 1:
            raise ValueError("state_dim must be a positive integer")
        if type(max_urgency) is not int or max_urgency < 1:
            raise ValueError("max_urgency must be a positive integer")
        if not all(
            0 <= p <= 1 for p in (survey_success, pickup_success, per_step_loss)
        ):
            raise ValueError("probabilities must be in [0, 1]")
        self.max_urgency = max_urgency
        self.survey_success = float(survey_success)
        self.pickup_success = float(pickup_success)
        self.per_step_loss = float(per_step_loss)
        self.zones = max(2, round((state_dim - 2) / (6 * (max_urgency + 1))) - 1)
        super().__init__(6 * (self.zones + 1) * (max_urgency + 1) + 2, 5)
        self.requested_state_dim = state_dim
        self.rescued, self.lost = self.state_dim - 2, self.state_dim - 1
        self.name = (
            f"{self.state_dim}_{self.action_dim}_search_and_rescue_1"
            f"_{self.zones}_{max_urgency}"
            f"_{survey_success}_{pickup_success}_{per_step_loss}"
        )

    def _encode(self, zone, status, onboard, urgency):
        return ((zone * 3 + status) * 2 + onboard) * (self.max_urgency + 1) + urgency

    def _build_model(self):
        t = TransitionBuilder(self.state_dim, self.action_dim)
        for zone in range(self.zones + 1):
            for status in range(3):
                for onboard in (0, 1):
                    for urgency in range(self.max_urgency + 1):
                        s = self._encode(zone, status, onboard, urgency)
                        for action in range(5):
                            t.rewards[s, action] = -1
                            # Rescue and death are both explicitly terminal.
                            if action == 4 and onboard:
                                t.add(action, s, self.rescued, 1 - self.per_step_loss)
                                t.add(action, s, self.lost, self.per_step_loss)
                                t.rewards[s, action] += 60 * (1 - self.per_step_loss)
                                t.rewards[s, action] -= 30 * self.per_step_loss
                                continue
                            if urgency == 0:
                                t.add(action, s, self.lost, 1)
                                t.rewards[s, action] -= 30
                                continue
                            if action == 0:
                                possibilities = [
                                    (min(self.zones, zone + 1), 0, onboard, 1.0)
                                ]
                            elif action == 1:
                                possibilities = [(max(0, zone - 1), 0, onboard, 1.0)]
                            elif action == 2 and zone > 0 and status == 0:
                                possibilities = [
                                    (zone, 1, onboard, self.survey_success),
                                    (zone, 2, onboard, 1 - self.survey_success),
                                ]
                            elif (
                                action == 3 and status == 1 and zone > 0 and not onboard
                            ):
                                possibilities = [
                                    (zone, status, 1, self.pickup_success),
                                    (zone, status, 0, 1 - self.pickup_success),
                                ]
                            elif action == 4:
                                possibilities = [(0, 0, onboard, 1.0)]
                            else:
                                possibilities = [(zone, status, onboard, 1.0)]
                                t.rewards[s, action] -= 2
                            for (
                                next_zone,
                                next_status,
                                next_onboard,
                                p,
                            ) in possibilities:
                                t.add(action, s, self.lost, p * self.per_step_loss)
                                t.add(
                                    action,
                                    s,
                                    self._encode(
                                        next_zone,
                                        next_status,
                                        next_onboard,
                                        urgency - 1,
                                    ),
                                    p * (1 - self.per_step_loss),
                                )
                            t.rewards[s, action] -= 30 * self.per_step_loss
        t.absorbing(self.rescued)
        t.absorbing(self.lost)
        self.transition_matrix, self.reward_matrix = t.finish(), t.rewards
