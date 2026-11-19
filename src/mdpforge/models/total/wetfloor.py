"""Wet-floor grid navigation: discounted adaptation of slippery-cell SSPs."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from mdpforge.core.mdp import MDP

TEST_PARAMETERS = {"state_dim": 16, "seed": 2}
METADATA = {
    "category": "navigation",
    "description": "Gridworld with seeded slippery cells and absorbing destination, inspired by WetFloor.",
    "reference": "Bonet and Geffner (2006), ICAPS; simplified discounted WetFloor adaptation, this implementation.",
    "sizes": {
        "tiny": {"requested_state_dim": 16, "parameters": {}},
        "small": {"requested_state_dim": 100, "parameters": {}},
        "medium": {"requested_state_dim": 400, "parameters": {}},
        "large": {"requested_state_dim": 1600, "parameters": {}},
    },
}


class Model(MDP):
    def __init__(
        self,
        state_dim: int = 100,
        action_dim: int = 4,
        *,
        slippery_fraction: float = 0.2,
        slip_probability: float = 0.35,
        seed: int = 0,
    ):
        if (
            isinstance(state_dim, bool)
            or not isinstance(state_dim, int)
            or state_dim < 4
        ):
            raise ValueError("state_dim must be integer >=4")
        if action_dim != 4:
            raise ValueError("WetFloor requires four actions")
        if not np.isfinite(slippery_fraction) or not 0 <= slippery_fraction <= 1:
            raise ValueError("slippery_fraction must be in [0,1]")
        if not np.isfinite(slip_probability) or not 0 <= slip_probability <= 1:
            raise ValueError("slip_probability must be in [0,1]")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.requested_state_dim = state_dim
        self.side = int(np.ceil(np.sqrt(state_dim)))
        self.state_dim = self.side**2
        self.action_dim = 4
        self.slippery_fraction = float(slippery_fraction)
        self.slip_probability = float(slip_probability)
        self.seed = seed
        self.terminal_states = (self.state_dim - 1,)
        self.name = f"{self.state_dim}_4_wetfloor_{self.side}_{slippery_fraction}_{slip_probability}_{seed}"

    def _build_model(self):
        n, width = self.state_dim, self.side
        directions = ((-1, 0), (0, 1), (1, 0), (0, -1))
        rng = np.random.default_rng(self.seed)
        slippery = rng.random(n) < self.slippery_fraction
        self.slippery_cells = slippery
        self.reward_matrix = -np.ones((n, 4), dtype=float)
        self.reward_matrix[-1, :] = 0
        matrices = []
        for action in range(4):
            rows, cols, data = [], [], []
            for s in range(n):
                if s == n - 1:
                    probs = {s: 1.0}
                else:
                    p = self.slip_probability if slippery[s] else 0
                    probs = {}
                    for d, weight in (
                        (action, 1 - p),
                        ((action + 1) % 4, p / 2),
                        ((action - 1) % 4, p / 2),
                    ):
                        dr, dc = directions[d]
                        r, c = divmod(s, width)
                        target = (
                            (r + dr) * width + c + dc
                            if 0 <= r + dr < width and 0 <= c + dc < width
                            else s
                        )
                        probs[target] = probs.get(target, 0) + weight
                for t, value in probs.items():
                    if value > 0:
                        rows.append(s)
                        cols.append(t)
                        data.append(value)
            matrices.append(csr_matrix((data, (rows, cols)), shape=(n, n)))
        self.transition_matrix = matrices
