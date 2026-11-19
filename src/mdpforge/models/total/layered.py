"""Layered SCC benchmark inspired by Dai et al. (2011)."""

from __future__ import annotations

from mdpforge.utils.topology_graphs import TopologyMDP

TEST_PARAMETERS = {"state_dim": 24, "action_dim": 3, "layers": 4, "seed": 3}
METADATA = {
    "category": "abstract",
    "description": "Layered MDP with connected recurrent blocks and directed dependencies between layers.",
    "reference": "Dai, Mausam, Weld and Goldsmith (2011), Topological Value Iteration Algorithms, JAIR 42.",
    "sizes": {
        "tiny": {"requested_state_dim": 24, "parameters": {"layers": 4}},
        "small": {"requested_state_dim": 120, "parameters": {"layers": 8}},
        "medium": {"requested_state_dim": 600, "parameters": {"layers": 24}},
        "large": {"requested_state_dim": 2400, "parameters": {"layers": 60}},
    },
}


class Model(TopologyMDP):
    def __init__(
        self,
        state_dim: int = 120,
        action_dim: int = 3,
        *,
        layers: int = 8,
        seed: int = 0,
    ):
        if (
            isinstance(layers, bool)
            or not isinstance(layers, int)
            or not (2 <= layers <= state_dim // 2)
        ):
            raise ValueError("layers must be between 2 and state_dim//2")
        self.layers = layers
        super().__init__(
            state_dim,
            action_dim,
            seed=seed,
            slug="layered",
            parameters=(layers,),
            goal_fraction=1 / state_dim,
        )

    def _adjacency(self, rng):
        n = self.state_dim
        # Divide states into balanced, disjoint consecutive layers.
        splits = [round(i * n / self.layers) for i in range(self.layers + 1)]
        graph = [set() for _ in range(n)]
        for layer in range(self.layers):
            start, stop = splits[layer : layer + 2]
            for s in range(start, stop):
                # Bidirectional ring makes each nonterminal layer one SCC.
                graph[s].add(start + (s - start + 1) % (stop - start))
                graph[s].add(start + (s - start - 1) % (stop - start))
                if layer + 1 < self.layers:
                    lo, hi = splits[layer + 1 : layer + 3]
                    graph[s].add(int(rng.integers(lo, hi)))
                else:
                    graph[s].add(n - 1)
        return graph
