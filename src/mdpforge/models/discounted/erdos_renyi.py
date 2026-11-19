"""Directed Erdős–Rényi graph with controllable edge probability."""

from __future__ import annotations

import numpy as np

from mdpforge.utils.topology_graphs import TopologyMDP

TEST_PARAMETERS = {"state_dim": 16, "action_dim": 3, "edge_probability": 0.2, "seed": 2}
METADATA = {
    "category": "abstract",
    "description": "Discounted goal-reaching MDP on a directed Erdos-Renyi G(n,p) graph.",
    "reference": "Erdos and Renyi (1959), On Random Graphs I; graph-to-MDP adaptation, this implementation.",
    "sizes": {
        "tiny": {"requested_state_dim": 16, "parameters": {"edge_probability": 0.2}},
        "small": {"requested_state_dim": 100, "parameters": {"edge_probability": 0.06}},
        "medium": {
            "requested_state_dim": 400,
            "parameters": {"edge_probability": 0.02},
        },
        "large": {
            "requested_state_dim": 1600,
            "parameters": {"edge_probability": 0.005},
        },
    },
}


class Model(TopologyMDP):
    def __init__(
        self,
        state_dim: int = 100,
        action_dim: int = 3,
        *,
        edge_probability: float = 0.06,
        seed: int = 0,
        goal_fraction: float = 0.05,
    ):
        if not np.isfinite(edge_probability) or not 0 <= edge_probability <= 1:
            raise ValueError("edge_probability must be in [0,1]")
        self.edge_probability = float(edge_probability)
        super().__init__(
            state_dim,
            action_dim,
            seed=seed,
            slug="erdosrenyi",
            parameters=(edge_probability,),
            goal_fraction=goal_fraction,
        )

    def _adjacency(self, rng):
        n = self.state_dim
        result = []
        for s in range(n):
            # Rows are independent Bernoulli trials; diagonal excluded.
            candidates = np.flatnonzero(rng.random(n) < self.edge_probability)
            result.append(set(int(x) for x in candidates if x != s))
        return result
