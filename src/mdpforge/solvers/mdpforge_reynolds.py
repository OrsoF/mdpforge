from collections import deque

import numpy as np
from scipy.sparse import csr_matrix

from mdpforge.core.mdp import MDP
from mdpforge.core.partition import Partition
from mdpforge.core.solve_result import SolveResult
from mdpforge.core.solver import GenericSolver

METADATA = {
    "display_name": "Reynolds · mdpforge",
    "family": "experimental",
    "backend": "mdpforge",
    "description": (
        "Reynolds-inspired Q-learning with transition-graph regions, "
        "uniform state sampling and reciprocal-visit learning rates."
    ),
}


class Solver(GenericSolver):
    def __init__(
        self,
        model: MDP,
        discount: float,
        final_precision: float = 1e-3,
        *,
        max_iterations: int = 100_000,
        initial_regions: int = 2,
        exploration_prob: float = 0.1,
        refine_interval: int = 1_000,
        split_threshold: float = 0.1,
        min_visits_before_split: int = 10,
    ):
        """Configure graph-based region Q-learning.

        final_precision: finite float > 0; external validation target, not a stop.
        max_iterations: int > 0; sampled transition budget.
        initial_regions: int > 0; initial graph groups, capped at the state count.
        exploration_prob: float in [0, 1]; probability of choosing a random action.
        refine_interval: int >= 0; updates between refinements; zero disables them.
        split_threshold: finite float >= 0; strict decision-loss threshold.
        min_visits_before_split: int > 0; common-action visits in both neighbors.
        """
        assert 0 < discount < 1
        for name, value, minimum in (
            ("max_iterations", max_iterations, 1),
            ("initial_regions", initial_regions, 1),
            ("refine_interval", refine_interval, 0),
            ("min_visits_before_split", min_visits_before_split, 1),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not np.isfinite(final_precision) or final_precision <= 0:
            raise ValueError("final_precision must be finite and positive")
        if not np.isfinite(exploration_prob) or not 0 <= exploration_prob <= 1:
            raise ValueError("exploration_prob must be in [0, 1]")
        if not np.isfinite(split_threshold) or split_threshold < 0:
            raise ValueError("split_threshold must be finite and nonnegative")
        super().__init__(
            model, discount, final_precision, max_iterations=max_iterations
        )
        self.initial_regions = initial_regions
        self.exploration_prob = exploration_prob
        self.refine_interval = refine_interval
        self.split_threshold = split_threshold
        self.min_visits_before_split = min_visits_before_split
        self.name = __name__.rsplit(".", 1)[-1]

    def _build_graph(self):
        """Build undirected adjacency from positive transitions of any action."""
        graph = csr_matrix((self.model.state_dim, self.model.state_dim), dtype=bool)
        for transition in self.model.transition_matrix:
            graph = graph.maximum(transition > 0)
        graph = graph.maximum(graph.T).tocsr()
        graph.setdiag(False)
        graph.eliminate_zeros()
        graph.sort_indices()
        return graph

    def _split_region(self, states):
        """Bisect a breadth-first graph ordering, traversing every component."""
        states = np.asarray(states, dtype=int)
        graph = self.graph[states][:, states]
        visited = np.zeros(len(states), dtype=bool)
        order = []
        roots = np.argsort(-np.diff(graph.indptr), kind="stable")
        for root in roots:
            if visited[root]:
                continue
            queue = deque([root])
            visited[root] = True
            while queue:
                state = queue.popleft()
                order.append(state)
                start, end = graph.indptr[state : state + 2]
                for neighbor in graph.indices[start:end]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
        ordered_states = states[order]
        midpoint = len(states) // 2
        return ordered_states[:midpoint], ordered_states[midpoint:]

    def _initial_partition(self):
        """Repeatedly bisect the largest graph group to bootstrap neighbors."""
        regions = [np.arange(self.model.state_dim)]
        while len(regions) < min(self.initial_regions, self.model.state_dim):
            largest = max(range(len(regions)), key=lambda index: len(regions[index]))
            left, right = self._split_region(regions[largest])
            regions[largest] = left
            regions.append(right)
        return Partition.from_regions(self.model, regions)

    def _refine_partition(self):
        """Split qualifying neighbors, inheriting Q and resetting child visits."""
        phi = self.partition.phi
        adjacency = (phi.T @ self.graph @ phi).tocoo()
        selected = set()
        q_value = self.contracted_q_value
        greedy = q_value.argmax(axis=1)
        for i, j in zip(adjacency.row, adjacency.col):
            if i >= j or greedy[i] == greedy[j]:
                continue
            enough_visits = np.any(
                (self.visits[i] >= self.min_visits_before_split)
                & (self.visits[j] >= self.min_visits_before_split)
            )
            loss_i = abs(q_value[i, greedy[i]] - q_value[i, greedy[j]])
            loss_j = abs(q_value[j, greedy[i]] - q_value[j, greedy[j]])
            if enough_visits and max(loss_i, loss_j) > self.split_threshold:
                selected.update((i, j))

        if not selected:
            return
        regions, parents, children = [], [], []
        for index, states in enumerate(self.partition.states_in_region):
            split = index in selected and len(states) > 1
            pieces = self._split_region(states) if split else (states,)
            for piece in pieces:
                regions.append(piece)
                parents.append(index)
                children.append(split)
        if not any(children):
            return
        self.contracted_q_value = q_value[parents]
        self.visits = self.visits[parents]
        self.visits[np.asarray(children)] = 0
        self.partition = Partition.from_regions(self.model, regions)

    def _solve(self) -> SolveResult:
        self.graph = self._build_graph()
        self.partition = self._initial_partition()
        shape = (self.partition.n_regions, self.model.action_dim)
        self.contracted_q_value = np.zeros(shape)
        self.visits = np.zeros(shape, dtype=np.int64)

        for iteration in range(1, self.max_iterations + 1):
            state = np.random.randint(self.model.state_dim)
            labels = self.partition.state_to_region
            region = labels[state]
            if np.random.random() < self.exploration_prob:
                action = np.random.randint(self.model.action_dim)
            else:
                action = self.contracted_q_value[region].argmax()
            transition = self.model.transition_matrix[action]
            start, end = transition.indptr[state : state + 2]
            next_state = np.random.choice(
                transition.indices[start:end], p=transition.data[start:end]
            )
            self.visits[region, action] += 1
            target = self.model.reward_matrix[state, action] + self.discount * (
                self.contracted_q_value[labels[next_state]].max()
            )
            delta = target - self.contracted_q_value[region, action]
            self.contracted_q_value[region, action] += (
                delta / self.visits[region, action]
            )
            if not np.isfinite(self.contracted_q_value[region, action]):
                raise RuntimeError("Reynolds stopped: non_finite values")
            if (
                self.refine_interval
                and iteration % self.refine_interval == 0
                and iteration < self.max_iterations
            ):
                self._refine_partition()

        self.n_regions = self.partition.n_regions
        self.q_value = self.contracted_q_value[self.partition.state_to_region]
        self.value = self.q_value.max(axis=1)
        self.policy = self.q_value.argmax(axis=1)
        return SolveResult.from_solver(self, validation_method="residual")
