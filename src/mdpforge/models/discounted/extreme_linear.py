"""Extreme linear SSP topologies adapted to discounted reward."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from mdpforge.core.mdp import MDP

TEST_PARAMETERS = {"state_dim": 12, "variant": 2}
METADATA = {
    "category": "abstract",
    "description": "Linear goal-reaching chains: directed deterministic, bidirectional deterministic, or bidirectional stochastic.",
    "reference": "Gravel and Champagne Gareau (2025), Topology-Driven Solver Selection, section 3; discounted adaptation.",
    "sizes": {
        "tiny": {"requested_state_dim": 16, "parameters": {"variant": 2}},
        "small": {"requested_state_dim": 100, "parameters": {"variant": 2}},
        "medium": {"requested_state_dim": 400, "parameters": {"variant": 0}},
        "large": {"requested_state_dim": 1600, "parameters": {"variant": 0}},
    },
}


class Model(MDP):
    def __init__(
        self,
        state_dim: int = 100,
        action_dim: int | None = None,
        *,
        variant: int = 2,
        forward_probability: float = 0.55,
    ):
        if (
            isinstance(state_dim, bool)
            or not isinstance(state_dim, int)
            or state_dim < 3
        ):
            raise ValueError("state_dim must be integer >=3")
        if variant not in (0, 1, 2):
            raise ValueError(
                "variant must be 0 (forward), 1 (bidirectional deterministic), or 2 (bidirectional stochastic)"
            )
        if not np.isfinite(forward_probability) or not (0 < forward_probability <= 1):
            raise ValueError("forward_probability must be in (0,1]")
        expected_actions = 2 if variant == 1 else 1
        if action_dim is not None and action_dim != expected_actions:
            raise ValueError(
                f"variant {variant} requires action_dim={expected_actions}"
            )
        self.requested_state_dim = self.state_dim = state_dim
        self.action_dim = expected_actions
        self.variant = int(variant)
        self.forward_probability = float(forward_probability)
        self.terminal_states = (state_dim - 1,)
        self.name = f"{state_dim}_{self.action_dim}_extremelinear_{variant}_{forward_probability}"

    def _build_model(self):
        n = self.state_dim
        self.reward_matrix = -np.ones((n, self.action_dim), dtype=float)
        self.reward_matrix[-1, :] = 0.0
        transitions = []
        for action in range(self.action_dim):
            rows, cols, data = [], [], []
            for s in range(n):
                if s == n - 1:
                    next_states = ((s, 1.0),)
                elif self.variant == 0:
                    next_states = ((s + 1, 1.0),)
                elif self.variant == 1:
                    next_states = (
                        (min(n - 1, s + 1) if action == 0 else max(0, s - 1), 1.0),
                    )
                else:
                    next_states = (
                        (s + 1, self.forward_probability),
                        (max(0, s - 1), 1 - self.forward_probability),
                    )
                for t, prob in next_states:
                    if prob > 0:
                        rows.append(s)
                        cols.append(t)
                        data.append(prob)
            transitions.append(csr_matrix((data, (rows, cols)), shape=(n, n)))
        self.transition_matrix = transitions
