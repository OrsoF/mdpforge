"""Extreme complete-graph SSP topologies adapted to discounted reward."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from mdpforge.core.mdp import MDP

TEST_PARAMETERS = {"state_dim": 8, "variant": 1}
METADATA = {
    "category": "abstract",
    "description": "Complete-graph benchmark: deterministic choice of each neighbor or one uniform stochastic action.",
    "reference": "Gravel and Champagne Gareau (2025), Topology-Driven Solver Selection, section 3; discounted adaptation.",
    "sizes": {
        "tiny": {"requested_state_dim": 8, "parameters": {"variant": 1}},
        "small": {"requested_state_dim": 20, "parameters": {"variant": 1}},
        "medium": {"requested_state_dim": 50, "parameters": {"variant": 1}},
        "large": {"requested_state_dim": 100, "parameters": {"variant": 1}},
    },
}


class Model(MDP):
    def __init__(
        self, state_dim: int = 20, action_dim: int | None = None, *, variant: int = 1
    ):
        if (
            isinstance(state_dim, bool)
            or not isinstance(state_dim, int)
            or state_dim < 3
        ):
            raise ValueError("state_dim must be integer >=3")
        if variant not in (0, 1):
            raise ValueError("variant must be 0 (deterministic) or 1 (stochastic)")
        actions = state_dim - 1 if variant == 0 else 1
        if action_dim is not None and action_dim != actions:
            raise ValueError(f"variant {variant} requires action_dim={actions}")
        self.requested_state_dim = self.state_dim = state_dim
        self.action_dim = actions
        self.variant = int(variant)
        self.terminal_states = (state_dim - 1,)
        self.name = f"{state_dim}_{actions}_extremedense_{variant}"

    def _build_model(self):
        n = self.state_dim
        self.reward_matrix = -np.ones((n, self.action_dim))
        self.reward_matrix[-1, :] = 0
        self.transition_matrix = []
        for a in range(self.action_dim):
            rows, cols, data = [], [], []
            for s in range(n):
                if s == n - 1:
                    targets = [(s, 1.0)]
                elif self.variant == 0:
                    # Enumerate every other state, including the goal.
                    t = a if a < s else a + 1
                    targets = [(t, 1.0)]
                else:
                    targets = [(t, 1 / (n - 1)) for t in range(n) if t != s]
                for t, p in targets:
                    rows.append(s)
                    cols.append(t)
                    data.append(p)
            self.transition_matrix.append(
                csr_matrix((data, (rows, cols)), shape=(n, n))
            )
