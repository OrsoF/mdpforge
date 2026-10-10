"""Certified progressive Bellman-envelope value iteration (precomputed transitions).

Compatible with mdpforge's MDPProtocol and Partition interfaces (October 2026).
For a partition with n states and K regions, precompute P_a @ phi once.
All envelope backups subsequently use *K-dimensional* vectors:

    L_P(x)[B] = min_{s in B} max_a (r_sa + gamma (P_a phi x)_s)
    U_P(x)[B] = max_{s in B} max_a (r_sa + gamma (P_a phi x)_s)

The returned lower_value and upper_value are valid pointwise bounds on V*.
With stop_criterion="envelope", value is the midpoint and
    certified_error = ||upper-lower||_inf / 2.
With stop_criterion="residual" (default), a ground-state VI polish also
    enforces ||T(value)-value||_inf/(1-gamma) <= final_precision,
which is needed for benchmark validators using the classical residual bound.
This extra polish may dominate the runtime on high-discount MDPs.
Only finite discounted MDPs with rewards r(s,a) and stochastic P_a apply.

The lower/upper bounds are *not* projected Bellman means. They cannot in
general be represented by a single ordinary aggregated MDP.
"""

from time import perf_counter

import numpy as np

from mdpforge.core.model import MDPProtocol
from mdpforge.core.operators import bellman_operator, compact_optimal_bellman_operator
from mdpforge.core.partition import Partition


class Solver:
    solver_type = "vi"

    def __init__(
        self,
        model: MDPProtocol,
        discount: float,
        final_precision: float = 1e-3,
        verbose: bool = False,
        refinement: str = "width",
        n_tiles: int | None = None,
        projected_steps: int = 100,
        refinement_width: float | None = None,
        switch_to_vi_ratio: float = 0.9,
        stop_criterion: str = "residual",
    ):
        if not 0 < discount < 1:
            raise ValueError("discount must belong to (0, 1)")
        if final_precision <= 0:
            raise ValueError("final_precision must be positive")
        if projected_steps < 1:
            raise ValueError("projected_steps must be >= 1")
        if refinement not in {"width", "tiles"}:
            raise ValueError("refinement must be 'width' or 'tiles'")
        if refinement_width is not None and refinement_width <= 0:
            raise ValueError("refinement_width must be positive")
        if n_tiles is not None and n_tiles < 2:
            raise ValueError("n_tiles must be >= 2")
        if not 0 < switch_to_vi_ratio <= 1:
            raise ValueError("switch_to_vi_ratio must belong to (0, 1]")
        if stop_criterion not in {"envelope", "residual"}:
            raise ValueError("stop_criterion must be 'envelope' or 'residual'")

        self.model = model
        self.discount = discount
        self.epsilon = final_precision
        self.verbose = verbose
        self.refinement = refinement
        self.n_tiles = n_tiles
        self.projected_steps = int(projected_steps)
        # A strict margin relative to 2 eps (1-gamma) ensures finite stopping.
        self.refinement_width = (
            final_precision * (1 - discount)
            if refinement_width is None
            else float(refinement_width)
        )
        if self.refinement_width >= 2 * final_precision * (1 - discount):
            raise ValueError(
                "refinement_width must be < 2 * final_precision * (1-discount) "
                "for the finite-termination guarantee"
            )
        self.switch_to_vi_ratio = switch_to_vi_ratio
        self.stop_criterion = stop_criterion
        self.shared_reward = np.all(
            model.reward_matrix == model.reward_matrix[:, [0]]
        )
        self.partition = Partition(model)
        self.name = "EnvelopeVItiles" if refinement == "tiles" else "EnvelopeVI"

        self.value = None
        self.policy = None
        self.lower_value = None
        self.upper_value = None
        self.certified_error = None
        self.residual_error_bound = None
        self.runtime = None
        self.n_envelope_updates = 0
        self.n_precomputations = 0
        self.n_refinements = 0
        self.n_ground_bellman_updates = 0
        self.precompute_time = 0.0
        self.region_history = []
        self.used_exact_vi = False

    def _precompute(self) -> None:
        """Build the K-column compressed ground-state transitions *once* per partition.

        mdpforge's aggregate_value_model computes [P_a @ phi], not the
        K-by-K averaged matrices used for Q-value aggregation. The distinction
        matters: min/max must be taken over *ground states* in each region.
        """
        start = perf_counter()
        transitions, rewards = self.partition.compute_agg_trans_reward_v()
        self._transitions = transitions  # list of CSR arrays, each shape (N, K)
        self._rewards = np.asarray(rewards, dtype=float)  # (N, A)
        self._labels = np.asarray(self.partition.state_to_region, dtype=int)
        self._K = self.partition.n_regions
        self._N = self.model.state_dim
        self._A = self.model.action_dim
        if self._rewards.shape != (self._N, self._A):
            raise ValueError("Expected reward_matrix with shape (N, A)")
        if len(transitions) != self._A or any(
            p.shape != (self._N, self._K) for p in transitions
        ):
            raise ValueError("Expected P_a @ phi of shape (N, K) for each action")

        # Cache grouping indices. NumPy's reduceat is substantially cheaper
        # than np.minimum.at/maximum.at for repeated grouped reductions.
        self._order = np.argsort(self._labels, kind="stable")
        counts = np.bincount(self._labels, minlength=self._K)
        if np.any(counts == 0):
            raise ValueError("Partition contains an empty region")
        self._starts = np.r_[0, np.cumsum(counts)[:-1]]
        self.n_precomputations += 1
        self.precompute_time += perf_counter() - start

    def _compressed_bellman(self, x: np.ndarray) -> np.ndarray:
        """Ground Bellman image T(phi x), with O(sum_a nnz(P_a phi)) work."""
        x = np.asarray(x, dtype=float)
        if x.shape != (self._K,):
            raise ValueError("Expected a region-value vector of length K")
        if self.shared_reward:
            y = np.asarray(self._transitions[0] @ x, dtype=float).ravel()
            for p in self._transitions[1:]:
                np.maximum(y, np.asarray(p @ x).ravel(), out=y)
            y *= self.discount
            y += self._rewards[:, 0]
        else:
            y = self._rewards[:, 0] + self.discount * np.asarray(
                self._transitions[0] @ x
            ).ravel()
            for a in range(1, self._A):
                z = self._rewards[:, a] + self.discount * np.asarray(
                    self._transitions[a] @ x
                ).ravel()
                np.maximum(y, z, out=y)
        return y

    def _reduce(self, ground_values: np.ndarray, *, lower: bool) -> np.ndarray:
        sorted_values = ground_values[self._order]
        if lower:
            return np.minimum.reduceat(sorted_values, self._starts)
        return np.maximum.reduceat(sorted_values, self._starts)

    def _envelope_step(self, x: np.ndarray, *, lower: bool) -> np.ndarray:
        return self._reduce(self._compressed_bellman(x), lower=lower)

    def _span(self, v: np.ndarray) -> float:
        sorted_values = v[self._order]
        small = np.minimum.reduceat(sorted_values, self._starts)
        large = np.maximum.reduceat(sorted_values, self._starts)
        return float(np.max(large - small))

    def _ground_bellman(self, value: np.ndarray) -> np.ndarray:
        self.n_ground_bellman_updates += 1
        return np.asarray(
            compact_optimal_bellman_operator(
                self.model, value, self.discount, self.shared_reward
            ),
            dtype=float,
        ).ravel()

    def _refine(self, signal: np.ndarray) -> bool:
        old_K = self.partition.n_regions
        if self.refinement == "tiles":
            tiles = self.n_tiles or max(2, int(self.model.state_dim ** 0.25))
            self.partition.refine_by_tiles(signal, tiles)
        else:
            self.partition.refine_by_width(signal, self.refinement_width)
        return self.partition.n_regions > old_K

    def _residual_polish(self, value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Force the *classical* residual bound below epsilon.

        The envelope certificate may already be <= epsilon even when
        ||Tv-v||/(1-gamma) is much larger. This is not a correctness failure:
        the two quantities are different bounds. Polish only if the benchmark
        requires the latter bound.

        A constant shift centers the signed Bellman residual and minimizes its
        sup norm among constant shifts, since T(v+c 1)=Tv+gamma*c 1.
        Re-check the shifted vector to avoid assuming exact floating arithmetic.
        """
        threshold = self.epsilon * (1 - self.discount)
        while True:
            updated = self._ground_bellman(value)
            r = updated - value
            if np.max(np.abs(r)) <= threshold:
                return value, r

            shift = (float(r.min()) + float(r.max())) / (2 * (1 - self.discount))
            if abs(shift) > 0:
                shifted = value + shift
                # The residual shift is exact mathematically; verify numerically.
                r_shifted = self._ground_bellman(shifted) - shifted
                if np.max(np.abs(r_shifted)) <= threshold:
                    return shifted, r_shifted

            # Standard VI contracts the residual at rate gamma, ensuring
            # eventual termination when threshold is above floating noise.
            value = updated

    def _finish(self, lo: np.ndarray, hi: np.ndarray, start: float):
        lo = np.asarray(lo, dtype=float).reshape(-1)
        hi = np.asarray(hi, dtype=float).reshape(-1)
        if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
            raise FloatingPointError("Nonfinite Bellman envelopes")
        if np.any(lo > hi + 1e-10):
            raise FloatingPointError("Invalid lower/upper Bellman envelopes")

        value = (lo + hi) / 2
        if self.stop_criterion == "residual":
            value, r = self._residual_polish(value)
            self.residual_error_bound = float(np.max(np.abs(r)) / (1 - self.discount))
            # Intersect the envelope with the independent signed-residual
            # certificate, preserving the enclosure of V*.
            r_lo = value + float(r.min()) / (1 - self.discount)
            r_hi = value + float(r.max()) / (1 - self.discount)
            lo = np.maximum(lo, r_lo)
            hi = np.minimum(hi, r_hi)
            if np.any(lo > hi + 1e-8):
                raise FloatingPointError("Inconsistent certified bounds; check P and rewards")
        else:
            self.residual_error_bound = None

        self.lower_value = lo
        self.upper_value = hi
        self.value = value
        # The estimate can differ from the envelope midpoint after polishing.
        # Its error is bounded by the furthest endpoint, not half the width.
        interval_error = float(np.max(np.maximum(np.abs(value - lo), np.abs(value - hi))))
        self.certified_error = (
            interval_error
            if self.residual_error_bound is None
            else min(interval_error, self.residual_error_bound)
        )
        if self.certified_error > self.epsilon * (1 + 1e-10):
            raise ArithmeticError("Solver failed to meet its precision certificate")
        self.policy = bellman_operator(
            self.model, self.value, self.discount
        ).argmax(axis=1)
        self.runtime = perf_counter() - start

    def _ground_fallback(self, lo: np.ndarray, hi: np.ndarray):
        """Certified fallback: full VI on both endpoints, not residual heuristics."""
        self.used_exact_vi = True
        while float(np.max(hi - lo)) > 2 * self.epsilon:
            lo = self._ground_bellman(lo)
            hi = self._ground_bellman(hi)
        return lo, hi

    def run(self):
        """Return when the certified sup-norm value error is <= final_precision."""
        start = perf_counter()
        n = self.model.state_dim
        r = np.asarray(self.model.reward_matrix, dtype=float)
        lo = np.full(n, float(np.min(r)) / (1 - self.discount))
        hi = np.full(n, float(np.max(r)) / (1 - self.discount))
        self.n_envelope_updates = 0
        self.n_precomputations = 0
        self.n_refinements = 0
        self.n_ground_bellman_updates = 0
        self.precompute_time = 0.0
        self.used_exact_vi = False
        self.region_history = []

        if float(np.max(hi - lo)) <= 2 * self.epsilon:
            self._finish(lo, hi, start)
            return
        if self.partition.n_regions >= self.switch_to_vi_ratio * n:
            lo, hi = self._ground_fallback(lo, hi)
            self._finish(lo, hi, start)
            return

        self._precompute()
        xlo = self._reduce(lo, lower=True)
        xhi = self._reduce(hi, lower=False)

        while True:
            for _ in range(self.projected_steps):
                # Region vectors only: the n-state value is never materialized
                # inside the repeated approximate Bellman updates.
                xlo = self._envelope_step(xlo, lower=True)
                xhi = self._envelope_step(xhi, lower=False)
                self.n_envelope_updates += 1
                gap = float(np.max(xhi - xlo))
                if gap <= 2 * self.epsilon:
                    self._finish(xlo[self._labels], xhi[self._labels], start)
                    return

            K = self._K
            self.region_history.append((K, gap))
            if self.verbose:
                print(f"K={K:5d} gap={gap:.3e}, certificate={gap/2:.3e}")

            if K >= self.switch_to_vi_ratio * n:
                ground_lo, ground_hi = self._ground_fallback(
                    xlo[self._labels], xhi[self._labels]
                )
                self._finish(ground_lo, ground_hi, start)
                return

            # Two cheap compressed ground Bellman images. Using only the gap
            # is invalid because it is constant within each block.
            tl = self._compressed_bellman(xlo)
            tu = self._compressed_bellman(xhi)
            sl, su = self._span(tl), self._span(tu)
            if max(sl, su) <= self.refinement_width:
                # Continue contraction on the same partition. Under the strict
                # threshold restriction in __init__, this cannot persist
                # forever while gap > 2*epsilon (in exact arithmetic).
                continue

            signal = tl if sl >= su else tu
            if not self._refine(signal):
                # Guard against rounding/tie behavior in external Partition.
                ground_lo, ground_hi = self._ground_fallback(
                    xlo[self._labels], xhi[self._labels]
                )
                self._finish(ground_lo, ground_hi, start)
                return

            self.n_refinements += 1
            # Refined regions are subsets of the old regions: previous bounds
            # remain certified and constant inside the new regions.
            ground_lo = xlo[self._labels]
            ground_hi = xhi[self._labels]
            # Avoid building a nearly-full compressed model only to abandon
            # it at the next batch check.
            if self.partition.n_regions >= self.switch_to_vi_ratio * n:
                ground_lo, ground_hi = self._ground_fallback(
                    ground_lo, ground_hi
                )
                self._finish(ground_lo, ground_hi, start)
                return
            self._precompute()
            xlo = self._reduce(ground_lo, lower=True)
            xhi = self._reduce(ground_hi, lower=False)
