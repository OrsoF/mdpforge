"""Watts–Strogatz ring lattice with independent clockwise edge rewiring."""

from __future__ import annotations

import numpy as np

from mdpforge.utils.topology_graphs import TopologyMDP

TEST_PARAMETERS = {
    "state_dim": 16,
    "action_dim": 3,
    "degree": 4,
    "rewiring_probability": 0.2,
    "seed": 2,
}
METADATA = {
    "category": "abstract",
    "description": "Discounted goal-reaching MDP on a Watts-Strogatz small-world graph.",
    "reference": "Watts and Strogatz (1998), Nature 393:440-442; graph-to-MDP adaptation, this implementation.",
    "sizes": {
        "tiny": {"requested_state_dim": 16, "parameters": {"degree": 4}},
        "small": {"requested_state_dim": 100, "parameters": {"degree": 6}},
        "medium": {"requested_state_dim": 400, "parameters": {"degree": 6}},
        "large": {"requested_state_dim": 1600, "parameters": {"degree": 8}},
    },
}


class Model(TopologyMDP):
    def __init__(
        self,
        state_dim: int = 100,
        action_dim: int = 3,
        *,
        degree: int = 6,
        rewiring_probability: float = 0.15,
        seed: int = 0,
        goal_fraction: float = 0.05,
    ):
        if (
            isinstance(degree, bool)
            or not isinstance(degree, int)
            or degree < 2
            or degree % 2
            or degree >= state_dim
        ):
            raise ValueError("degree must be even, >=2 and <state_dim")
        if not np.isfinite(rewiring_probability) or not 0 <= rewiring_probability <= 1:
            raise ValueError("rewiring_probability must lie in [0,1]")
        self.degree = degree
        self.rewiring_probability = float(rewiring_probability)
        super().__init__(
            state_dim,
            action_dim,
            seed=seed,
            slug="wattsstrogatz",
            parameters=(degree, rewiring_probability),
            goal_fraction=goal_fraction,
        )

    def _adjacency(self, rng):
        n = self.state_dim
        neighbors = [set() for _ in range(n)]
        # Standard undirected ring lattice (degree k), prior to rewiring.
        for s in range(n):
            for distance in range(1, self.degree // 2 + 1):
                t = (s + distance) % n
                neighbors[s].add(t)
                neighbors[t].add(s)
        for distance in range(1, self.degree // 2 + 1):
            for s in range(n):
                t = (s + distance) % n
                if rng.random() >= self.rewiring_probability:
                    continue
                # Replace the original edge while preserving its source.
                neighbors[s].remove(t)
                neighbors[t].remove(s)
                candidates = [v for v in range(n) if v != s and v not in neighbors[s]]
                new_target = int(rng.choice(candidates))
                neighbors[s].add(new_target)
                neighbors[new_target].add(s)
        return neighbors
