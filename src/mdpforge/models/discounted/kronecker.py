"""Row-normalized Kronecker product sampling, sparsified by outgoing samples."""

from __future__ import annotations

import numpy as np

from mdpforge.utils.topology_graphs import TopologyMDP

TEST_PARAMETERS = {"state_dim": 16, "action_dim": 3, "out_degree": 4, "seed": 2}
METADATA = {
    "category": "abstract",
    "description": "Discounted graph MDP with sparse samples from a Kronecker-product kernel (powers of two only).",
    "reference": "Leskovec et al. (2010), Kronecker Graphs, JMLR 11; sampled row-kernel adaptation, this implementation.",
    "sizes": {
        "tiny": {"requested_state_dim": 16, "parameters": {"out_degree": 4}},
        "small": {"requested_state_dim": 128, "parameters": {"out_degree": 8}},
        "medium": {"requested_state_dim": 512, "parameters": {"out_degree": 8}},
        "large": {"requested_state_dim": 2048, "parameters": {"out_degree": 12}},
    },
}


class Model(TopologyMDP):
    def __init__(
        self,
        state_dim: int = 128,
        action_dim: int = 3,
        *,
        out_degree: int = 8,
        seed: int = 0,
        goal_fraction: float = 0.05,
    ):
        if state_dim < 4 or state_dim & (state_dim - 1):
            raise ValueError("Kronecker state_dim must be a power of two >= 4")
        if (
            isinstance(out_degree, bool)
            or not isinstance(out_degree, int)
            or out_degree < 1
        ):
            raise ValueError("out_degree must be a positive integer")
        self.out_degree = out_degree
        super().__init__(
            state_dim,
            action_dim,
            seed=seed,
            slug="kronecker",
            parameters=(out_degree,),
            goal_fraction=goal_fraction,
        )

    def _adjacency(self, rng):
        # K = [[0.9, 0.5], [0.5, 0.1]]. Conditional on a source's
        # bits, target bits are independent, with P(target_bit | source_bit)
        # proportional to K[source_bit, target_bit]. This creates a sparse
        # approximation of the *row-normalized* Kronecker kernel.
        kernel = np.array([[0.9, 0.5], [0.5, 0.1]], dtype=float)
        kernel /= kernel.sum(axis=1, keepdims=True)
        bits = self.state_dim.bit_length() - 1
        graph = []
        for s in range(self.state_dim):
            neighbors = set()
            for _ in range(self.out_degree):
                t = 0
                for bit in range(bits):
                    source_bit = (s >> bit) & 1
                    if rng.random() < kernel[source_bit, 1]:
                        t |= 1 << bit
                if t != s:
                    neighbors.add(t)
            graph.append(neighbors)
        return graph
