from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix


class Partition:
    """Partition of ``n_states`` represented by contiguous region labels."""

    def __init__(self, model, labels=None):
        self.model = model
        n = model.state_dim
        self._labels = (
            np.zeros(n, dtype=int)
            if labels is None
            else self._normalize_labels(np.asarray(labels, dtype=int))
        )
        self._clear_cache()

    @staticmethod
    def _normalize_labels(labels: np.ndarray) -> np.ndarray:
        _, labels = np.unique(labels, return_inverse=True)
        return labels

    @classmethod
    def from_regions(cls, model, regions):
        labels = np.empty(model.state_dim, dtype=int)
        for k, region in enumerate(regions):
            labels[np.asarray(region, dtype=int)] = k
        return cls(model, labels)

    @classmethod
    def from_value_bins(cls, model, value: np.ndarray, n_regions: int):
        n_regions = int(np.clip(n_regions, 1, model.state_dim))
        if np.ptp(value) == 0:
            return cls(model)
        edges = np.linspace(value.min(), value.max(), n_regions + 1)[1:-1]
        return cls(model, np.searchsorted(edges, value))

    @classmethod
    def random(cls, model, n_regions: int, rng: np.random.Generator):
        n_regions = int(np.clip(n_regions, 1, model.state_dim))
        labels = np.r_[
            np.arange(n_regions),
            rng.integers(n_regions, size=model.state_dim - n_regions),
        ]
        rng.shuffle(labels)
        return cls(model, labels)

    def _clear_cache(self):
        for name in (
            "_phi",
            "_weights",
            "_states_in_region",
            "_aggregate_kind",
            "aggregate_transition_matrix",
            "aggregate_reward_matrix",
            "aggregate_transition_policy",
            "aggregate_reward_policy",
        ):
            self.__dict__.pop(name, None)

    @property
    def n_regions(self) -> int:
        return int(self._labels.max()) + 1

    @property
    def state_to_region(self) -> np.ndarray:
        return self._labels

    @property
    def states_in_region(self) -> list[list[int]]:
        if not hasattr(self, "_states_in_region"):
            order = np.argsort(self._labels, kind="stable")
            cuts = np.flatnonzero(np.diff(self._labels[order])) + 1
            self._states_in_region = [x.tolist() for x in np.split(order, cuts)]
        return self._states_in_region

    @states_in_region.setter
    def states_in_region(self, regions):
        labels = np.empty(self.model.state_dim, dtype=int)
        for k, region in enumerate(regions):
            labels[np.asarray(region, dtype=int)] = k
        self._labels = self._normalize_labels(labels)
        self._clear_cache()

    @property
    def phi(self) -> csr_matrix:
        if not hasattr(self, "_phi"):
            rows = np.arange(self.model.state_dim)
            self._phi = csr_matrix(
                (np.ones(self.model.state_dim), (rows, self._labels)),
                shape=(self.model.state_dim, self.n_regions),
            )
        return self._phi

    @property
    def weights(self) -> csr_matrix:
        if not hasattr(self, "_weights"):
            sizes = np.bincount(self._labels, minlength=self.n_regions)
            self._weights = self.phi.T.multiply(1 / sizes[:, None]).tocsr()
        return self._weights

    def span(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value)
        shape = (self.n_regions,) + value.shape[1:]
        lo = np.full(shape, np.inf)
        hi = np.full(shape, -np.inf)
        np.minimum.at(lo, self._labels, value)
        np.maximum.at(hi, self._labels, value)
        return hi - lo

    def refine_by_width(self, value: np.ndarray, epsilon: float):
        value = np.asarray(value)
        values = value[:, None] if value.ndim == 1 else value
        lo = np.full((self.n_regions, values.shape[1]), np.inf)
        np.minimum.at(lo, self._labels, values)
        bins = np.floor((values - lo[self._labels]) / epsilon).astype(np.int64)
        keys = tuple(bins[:, j] for j in range(bins.shape[1] - 1, -1, -1))
        order = np.lexsort(keys + (self._labels,))
        old = self._labels[order]
        sorted_bins = bins[order]
        new_group = np.r_[
            True,
            (old[1:] != old[:-1]) | np.any(sorted_bins[1:] != sorted_bins[:-1], axis=1),
        ]
        labels = np.empty_like(self._labels)
        labels[order] = np.cumsum(new_group) - 1
        parent = old[new_group]
        if np.array_equal(labels, self._labels):
            return parent
        self._labels = labels
        self._clear_cache()
        return parent

    def refine_by_tiles(self, value: np.ndarray, n_tiles: int):
        labels = np.empty(self.model.state_dim, dtype=int)
        parent = []
        k = 0
        for old_k, region in enumerate(self.states_in_region):
            region = np.asarray(region)
            sorted_region = region[np.argsort(value[region])]
            for tile in np.array_split(sorted_region, min(n_tiles, len(region))):
                labels[tile] = k
                parent.append(old_k)
                k += 1
        self._labels = labels
        self._clear_cache()
        return np.asarray(parent)

    def refine_region(self, region_index: int, pieces):
        regions = self.states_in_region
        regions[region_index] = list(pieces[0])
        regions.extend(map(list, pieces[1:]))
        parent = np.r_[
            np.arange(self.n_regions), np.repeat(region_index, len(pieces) - 1)
        ]
        self.states_in_region = regions
        return parent

    def compute_agg_trans_reward_v(self):
        if getattr(self, "_aggregate_kind", None) == "v":
            return self.aggregate_transition_matrix, self.aggregate_reward_matrix
        (
            self.aggregate_transition_matrix,
            self.aggregate_reward_matrix,
        ) = aggregate_value_model(self.model, self)
        self._aggregate_kind = "v"
        return self.aggregate_transition_matrix, self.aggregate_reward_matrix

    def compute_agg_trans_reward_q(self):
        if getattr(self, "_aggregate_kind", None) == "q":
            return self.aggregate_transition_matrix, self.aggregate_reward_matrix
        (
            self.aggregate_transition_matrix,
            self.aggregate_reward_matrix,
        ) = aggregate_q_model(self.model, self)
        self._aggregate_kind = "q"
        return self.aggregate_transition_matrix, self.aggregate_reward_matrix

    def compute_agg_trans_reward_pi(self, transition_policy, reward_policy):
        (
            self.aggregate_transition_policy,
            self.aggregate_reward_policy,
        ) = aggregate_policy(transition_policy, reward_policy, self)
        return self.aggregate_transition_policy, self.aggregate_reward_policy


def aggregate_value_model(model, partition: Partition):
    phi = partition.phi
    transitions = [transition @ phi for transition in model.transition_matrix]
    return transitions, model.reward_matrix


def aggregate_q_model(model, partition: Partition):
    phi, weights = partition.phi, partition.weights
    return (
        [weights @ transition @ phi for transition in model.transition_matrix],
        weights @ model.reward_matrix,
    )


def aggregate_policy(transition, reward: np.ndarray, partition: Partition):
    return partition.weights @ transition @ partition.phi, partition.weights @ reward
