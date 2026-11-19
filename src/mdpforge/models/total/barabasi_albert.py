"""Preferential-attachment graph MDP."""

from __future__ import annotations

from mdpforge.utils.topology_graphs import TopologyMDP

TEST_PARAMETERS = {"state_dim": 16, "action_dim": 3, "attachment_count": 2, "seed": 2}
METADATA = {
    "category": "abstract",
    "description": "Discounted goal-reaching MDP on a preferential-attachment graph with hubs.",
    "reference": "Barabasi and Albert (1999), Science 286:509-512; graph-to-MDP adaptation, this implementation.",
    "sizes": {
        "tiny": {"requested_state_dim": 16, "parameters": {"attachment_count": 2}},
        "small": {"requested_state_dim": 100, "parameters": {"attachment_count": 3}},
        "medium": {"requested_state_dim": 400, "parameters": {"attachment_count": 3}},
        "large": {"requested_state_dim": 1600, "parameters": {"attachment_count": 4}},
    },
}


class Model(TopologyMDP):
    def __init__(
        self,
        state_dim: int = 100,
        action_dim: int = 3,
        *,
        attachment_count: int = 3,
        seed: int = 0,
        goal_fraction: float = 0.05,
    ):
        if (
            isinstance(attachment_count, bool)
            or not isinstance(attachment_count, int)
            or not (1 <= attachment_count < state_dim)
        ):
            raise ValueError("attachment_count must be in [1,state_dim)")
        self.attachment_count = attachment_count
        super().__init__(
            state_dim,
            action_dim,
            seed=seed,
            slug="barabasialbert",
            parameters=(attachment_count,),
            goal_fraction=goal_fraction,
        )

    def _adjacency(self, rng):
        n, m = self.state_dim, self.attachment_count
        graph = [set() for _ in range(n)]
        # Start with a clique of m+1 vertices (all degrees positive).
        pool = []
        for s in range(m + 1):
            for t in range(s):
                graph[s].add(t)
                graph[t].add(s)
                pool.extend((s, t))
        for s in range(m + 1, n):
            chosen = set()
            # Exact degree-proportional sampling *without replacement*.
            while len(chosen) < m:
                chosen.add(int(rng.choice(pool)))
            for t in chosen:
                graph[s].add(t)
                graph[t].add(s)
                pool.extend((s, t))
        return graph
