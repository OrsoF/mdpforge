"""Shared primitives for stochastic-shortest-path solvers (undiscounted total reward).

A safe, intentionally restricted SSP class is supported:
* rewards are negative of strictly positive costs at every nonterminal (s,a);
* terminal states absorb with zero reward for every action;
* every nonterminal state has a positive-probability path to a terminal.

These assumptions guarantee existence of a proper stationary policy and ensure
that an improper policy earns -infinity (equivalently infinite expected cost).
Bellman backups start from a valid *upper* bound (zero). Policy evaluation
provides a *lower* bound, giving a certified absolute-error gap even at gamma=1.
"""

from __future__ import annotations

import warnings
from collections import deque

import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from mdpforge.core.solve_result import SolveResult


class SSP:
    def __init__(self, model, precision: float, max_iterations: int):
        if not np.isfinite(precision) or precision <= 0:
            raise ValueError("final_precision must be finite and strictly positive")
        if type(max_iterations) is not int or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        self.model = model
        self.n = int(model.state_dim)
        self.na = int(model.action_dim)
        self.precision = float(precision)
        self.transitions = [p.tocsr() for p in model.transition_matrix]
        self.rewards = np.asarray(model.reward_matrix, dtype=float)
        self.goals = frozenset(int(s) for s in getattr(model, "terminal_states", ()))
        if not self.goals or any(s < 0 or s >= self.n for s in self.goals):
            raise ValueError("SSP model must declare nonempty valid terminal_states")
        if (
            self.rewards.shape != (self.n, self.na)
            or not np.isfinite(self.rewards).all()
        ):
            raise ValueError("reward_matrix must have shape (S,A), with finite entries")
        if len(self.transitions) != self.na:
            raise ValueError(
                "transition_matrix must contain one (S,S) matrix per action"
            )
        for p in self.transitions:
            if p.shape != (self.n, self.n) or not np.isfinite(p.data).all():
                raise ValueError("invalid transition matrix")
            if (p.data < 0).any() or not np.allclose(
                np.asarray(p.sum(axis=1)).ravel(), 1.0, atol=1e-10, rtol=0
            ):
                raise ValueError("each action needs a row-stochastic transition matrix")
        self.transient = np.array(
            [s for s in range(self.n) if s not in self.goals], dtype=int
        )
        for g in self.goals:
            if not np.allclose(self.rewards[g], 0, atol=1e-12, rtol=0):
                raise ValueError(
                    "terminal states must have zero reward for every action"
                )
            for p in self.transitions:
                a, b = p.indptr[g], p.indptr[g + 1]
                ind, weights = p.indices[a:b], p.data[a:b]
                if np.any((ind != g) & (weights > 1e-14)) or not np.isclose(
                    weights[ind == g].sum(), 1, atol=1e-10
                ):
                    raise ValueError(
                        "terminal states must be absorbing under every action"
                    )
        if len(self.transient) and not np.all(self.rewards[self.transient] < 0):
            raise ValueError(
                "SSP solvers require strictly negative nonterminal rewards (positive step costs)"
            )
        self.predecessors = [set() for _ in range(self.n)]
        self.succ = [[None] * self.na for _ in range(self.n)]
        for action, p in enumerate(self.transitions):
            for s in range(self.n):
                a, b = p.indptr[s], p.indptr[s + 1]
                successors = p.indices[a:b][p.data[a:b] > 0]
                weights = p.data[a:b][p.data[a:b] > 0]
                self.succ[s][action] = (successors, weights / weights.sum())
                for t in successors:
                    self.predecessors[int(t)].add(s)
        self.proper_policy = self._build_proper_policy()
        self.lower_policy = self.proper_policy.copy()
        self.lower = self.evaluate(self.lower_policy)
        self.upper = np.zeros(self.n)
        self.n_policy_evaluations = 1
        self.n_backups = 0

    def _build_proper_policy(self):
        """Reverse BFS: each chosen action has a successor at smaller rank."""
        reached = set(self.goals)
        frontier = deque(sorted(self.goals))
        policy = np.zeros(self.n, dtype=int)
        # Need action information; find a positive edge from an unreached s to t.
        while frontier:
            t = frontier.popleft()
            for s in self.predecessors[t]:
                if s not in reached:
                    a = next(
                        a for a, (targets, _) in enumerate(self.succ[s]) if t in targets
                    )
                    policy[s] = a
                    reached.add(s)
                    frontier.append(s)
        if len(reached) != self.n:
            raise ValueError("not all states can reach a terminal under some policy")
        return policy

    def is_proper(self, policy):
        reached = set(self.goals)
        predecessors = [[] for _ in range(self.n)]
        for s in self.transient:
            targets, _ = self.succ[int(s)][int(policy[s])]
            for t in targets:
                predecessors[int(t)].append(int(s))
        todo = list(self.goals)
        for t in todo:
            for s in predecessors[t]:
                if s not in reached:
                    reached.add(s)
                    todo.append(s)
        return len(reached) == self.n

    def evaluate(self, policy):
        if not self.is_proper(policy):
            raise ValueError("cannot evaluate an improper policy in a cyclic SSP")
        result = np.zeros(self.n)
        if not len(self.transient):
            return result
        # Efficient sparse P_pi assembly (one row at a time; full S -> then restrict).
        rows, cols, data = [], [], []
        for s in self.transient:
            indices, probabilities = self.succ[int(s)][int(policy[s])]
            for t, pr in zip(indices, probabilities):
                if t not in self.goals:
                    rows.append(int(s))
                    cols.append(int(t))
                    data.append(float(pr))
        P = csr_matrix((data, (rows, cols)), shape=(self.n, self.n))
        PP = P[self.transient, :][:, self.transient]
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            vals = spsolve(
                eye(len(self.transient), format="csr") - PP,
                self.rewards[self.transient, policy[self.transient]],
            )
        result[self.transient] = np.atleast_1d(vals)
        if not np.isfinite(result).all():
            raise RuntimeError("policy evaluation returned non-finite values")
        return result

    def q(self, value):
        return np.column_stack(
            [self.rewards[:, a] + self.transitions[a] @ value for a in range(self.na)]
        )

    def backup(self, s, value, allowed=None):
        if s in self.goals:
            return 0.0
        options = np.flatnonzero(allowed[s]) if allowed is not None else range(self.na)
        return max(
            self.rewards[s, a] + self.transitions[a].getrow(s).dot(value)[0]
            for a in options
        )

    def greedy(self, value):
        policy = np.asarray(self.q(value).argmax(axis=1), dtype=int)
        for g in self.goals:
            policy[g] = 0
        return policy

    def improve_lower(self):
        """Policy improvement preserves properness under strictly positive costs."""
        q = self.q(self.lower)
        policy = np.asarray(q.argmax(axis=1), dtype=int)
        keep = (
            q[np.arange(self.n), policy]
            <= q[np.arange(self.n), self.lower_policy] + 1e-11
        )
        policy[keep] = self.lower_policy[keep]
        if np.array_equal(policy, self.lower_policy):
            return
        if not self.is_proper(policy):
            raise RuntimeError(
                "policy improvement became improper; check SSP assumptions"
            )
        new_lower = self.evaluate(policy)
        if np.any(new_lower < self.lower - 1e-7):
            raise RuntimeError("policy improvement unexpectedly decreased a value")
        self.lower_policy = policy
        self.lower = new_lower
        self.n_policy_evaluations += 1

    def certify(self, upper, *, update_lower=True, iterations=None):
        """Return a certified result iff lower <= V* <= upper with gap <= epsilon."""
        self.upper = upper
        if update_lower:
            self.improve_lower()
        gap = upper - self.lower
        if np.min(gap) < -max(1e-8, 1e-10 * np.max(np.abs(self.lower))):
            raise RuntimeError(
                "upper/lower SSP bounds crossed; likely an invalid model"
            )
        if np.max(gap) <= self.precision:
            # Returning the value of a proper policy, not an unachievable pointwise bound.
            return SolveResult(
                self.lower.copy(),
                self.lower_policy.copy(),
                validation_method="residual",
                lower_value=self.lower.copy(),
                upper_value=upper.copy(),
                termination_reason="ssp_policy_gap",
                iterations=iterations,
            )
        return None

    def components_in_dependency_order(self, allowed=None):
        """Compute SCCs on s -> successors graph, returning sinks first."""
        rows, cols = [], []
        for s in range(self.n):
            if s in self.goals:
                continue
            for a in range(self.na):
                if allowed is not None and not allowed[s, a]:
                    continue
                for t in self.succ[s][a][0]:
                    rows.append(s)
                    cols.append(int(t))
        graph = csr_matrix(
            (np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(self.n, self.n)
        )
        count, group = connected_components(graph, directed=True, connection="strong")
        parts = [[] for _ in range(count)]
        for state, c in enumerate(group):
            parts[int(c)].append(state)
        dep = [set() for _ in range(count)]
        parent = [set() for _ in range(count)]
        for s, t in zip(rows, cols):
            i, j = int(group[s]), int(group[t])
            if i != j:
                dep[i].add(j)
                parent[j].add(i)
        todo = deque(c for c in range(count) if not dep[c])
        result = []
        while todo:
            c = todo.popleft()
            result.append([s for s in parts[c] if s not in self.goals])
            for p in parent[c]:
                dep[p].remove(c)
                if not dep[p]:
                    todo.append(p)
        assert len(result) == count
        return result


def validate_extra(name, value, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


class SSPBase:
    """MDPForge `total` solver interface: Solver(model, final_precision, ...)."""

    def __init__(self, model, final_precision=1e-3, *, max_iterations=100_000):
        self.model = model
        self.final_precision = final_precision
        self.max_iterations = max_iterations
        self.name = self.__class__.__module__.rsplit(".", 1)[-1]
        self.problem = SSP(model, final_precision, max_iterations)
        self.n_backups = 0
        self.n_policy_evaluations = 0

    def finish(self, answer):
        self.n_backups = self.problem.n_backups
        self.n_policy_evaluations = self.problem.n_policy_evaluations
        return answer


def solve_topologically(
    problem, upper, *, max_iterations, allowed=None, certify_every=1, local_sweeps=1000
):
    """TVI sweep: solve each SCC in reverse dependency order; certify globally."""
    components = problem.components_in_dependency_order(allowed)
    steps = 0
    for iteration in range(1, max_iterations + 1):
        for component in components:
            if not component:
                continue
            for _ in range(local_sweeps):
                delta = 0.0
                for s in component:
                    v = problem.backup(s, upper, allowed)
                    # Valid upper bounds are monotonically nonincreasing.
                    v = min(v, upper[s])
                    delta = max(delta, upper[s] - v)
                    upper[s] = v
                    problem.n_backups += 1
                if delta <= problem.precision * 0.1:
                    break
        steps += 1
        if steps % certify_every == 0:
            result = problem.certify(upper, iterations=steps)
            if result is not None:
                return result
    raise RuntimeError("SSP TVI: max_iterations reached without a certified value gap")
