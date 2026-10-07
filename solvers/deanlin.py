from time import time
from typing import Dict, List, Set, Tuple

import numpy as np

from core.model import GenericModel
from core.partition import Partition


class Solver:
    """Dean-Lin hierarchical policy construction (1995), reward version."""

    def __init__(
        self,
        model: GenericModel,
        discount: float,
        final_precision: float,
        verbose: bool = False,
        kappa: float = 100.0,
        local_precision: float | None = None,
        abstract_precision: float | None = None,
        max_local_steps: int = 10_000,
        max_abstract_steps: int = 10_000,
    ):
        self.model = model
        self.discount = discount
        self.epsilon = final_precision
        self.verbose = verbose
        self.kappa = kappa
        self.local_precision = local_precision or final_precision
        self.abstract_precision = (
            abstract_precision
            if abstract_precision is not None
            else final_precision * (1.0 - discount) / 10.0
        )
        self.max_local_steps = max_local_steps
        self.max_abstract_steps = max_abstract_steps
        self.name = "DeanLinHPC"
        self.partition = Partition(model)

        self.value = None
        self.policy = None
        self.abstract_value = None
        self.abstract_policy = None
        self.runtime = None
        self.steps = 0

    def run(self):
        start = time()
        self._prepare_regions()
        self._compute_region_graph()

        local_policies = self._compute_all_local_policies()
        abstract_model = self._build_abstract_mdp(local_policies)
        self.abstract_value, self.abstract_policy = self._solve_abstract_mdp(
            abstract_model
        )

        self.policy = self._glue_policy(local_policies)
        self.value = self._evaluate_global_policy(self.policy)
        self.runtime = time() - start
        return self

    def _prepare_regions(self):
        self.regions: List[np.ndarray] = [
            np.asarray(region, dtype=int) for region in self.partition.states_in_region
        ]
        self.n_regions = len(self.regions)

        self.state_to_region = np.empty(self.model.state_dim, dtype=int)
        for r, states in enumerate(self.regions):
            self.state_to_region[states] = r

        self.region_sets: List[Set[int]] = [
            set(map(int, states)) for states in self.regions
        ]
        self.local_index = [
            {int(s): i for i, s in enumerate(states)} for states in self.regions
        ]

    def _compute_region_graph(self):
        self.boundary = [set() for _ in range(self.n_regions)]
        self.periphery = [set() for _ in range(self.n_regions)]
        self.neighbors = [set() for _ in range(self.n_regions)]

        for r, states in enumerate(self.regions):
            region_set = self.region_sets[r]
            for s in states:
                for a in range(self.model.action_dim):
                    row = self.model.transition_matrix[a].getrow(s)
                    for s2, p in zip(row.indices, row.data):
                        if p > 0 and s2 not in region_set:
                            self.boundary[r].add(int(s))
                            self.periphery[r].add(int(s2))
                            self.neighbors[r].add(int(self.state_to_region[s2]))

        # Practical fallback for closed regions.
        for r in range(self.n_regions):
            if not self.neighbors[r]:
                self.neighbors[r].add(r)

    def _compute_all_local_policies(self):
        policies: Dict[Tuple[int, int], np.ndarray] = {}
        for r in range(self.n_regions):
            for target in sorted(self.neighbors[r]):
                policies[(r, target)] = (
                    self._solve_closed_region(r)
                    if target == r
                    else self._solve_local_policy(r, target)
                )
        return policies

    def _solve_local_policy(self, region: int, target_region: int):
        """Compute pi_{R->S} with beta=0 toward S and -kappa elsewhere."""
        states = self.regions[region]
        idx = self.local_index[region]
        region_set = self.region_sets[region]
        target_set = self.region_sets[target_region]

        v = np.zeros(len(states))
        policy_local = np.zeros(len(states), dtype=int)

        for _ in range(self.max_local_steps):
            old_v = v.copy()

            for i, s in enumerate(states):
                q = np.empty(self.model.action_dim)
                for a in range(self.model.action_dim):
                    continuation = 0.0
                    row = self.model.transition_matrix[a].getrow(s)

                    for s2, p in zip(row.indices, row.data):
                        s2 = int(s2)
                        if s2 in region_set:
                            continuation += p * old_v[idx[s2]]
                        else:
                            continuation += p * (
                                0.0 if s2 in target_set else -self.kappa
                            )

                    # Dean-Lin local problem is stochastic shortest path:
                    # no discount factor is used here.
                    q[a] = self.model.reward_matrix[s, a] + continuation

                policy_local[i] = int(np.argmax(q))
                v[i] = q[policy_local[i]]

            if np.max(np.abs(v - old_v)) < self.local_precision:
                break

        policy = np.zeros(self.model.state_dim, dtype=int)
        policy[states] = policy_local
        return policy

    def _solve_closed_region(self, region: int):
        states = self.regions[region]
        idx = self.local_index[region]
        region_set = self.region_sets[region]
        v = np.zeros(len(states))
        policy_local = np.zeros(len(states), dtype=int)

        for _ in range(self.max_local_steps):
            old_v = v.copy()
            for i, s in enumerate(states):
                q = np.empty(self.model.action_dim)
                for a in range(self.model.action_dim):
                    row = self.model.transition_matrix[a].getrow(s)
                    continuation = sum(
                        p * old_v[idx[int(s2)]]
                        for s2, p in zip(row.indices, row.data)
                        if int(s2) in region_set
                    )
                    q[a] = self.model.reward_matrix[s, a] + self.discount * continuation
                policy_local[i] = int(np.argmax(q))
                v[i] = q[policy_local[i]]

            if np.max(np.abs(v - old_v)) < self.local_precision:
                break

        policy = np.zeros(self.model.state_dim, dtype=int)
        policy[states] = policy_local
        return policy

    def _build_abstract_mdp(self, local_policies):
        actions = [sorted(neighbors) for neighbors in self.neighbors]
        max_actions = max(map(len, actions))

        # Destination-dependent p'_{RS}(pi) and r'_{RS}(pi).
        P = np.zeros((max_actions, self.n_regions, self.n_regions))
        R = np.zeros_like(P)

        for r in range(self.n_regions):
            starts = sorted(self.boundary[r]) or list(map(int, self.regions[r]))

            for a_abs, target in enumerate(actions[r]):
                policy = local_policies[(r, target)]
                P[a_abs, r] = self._region_exit_probabilities(r, policy, starts)
                R[a_abs, r] = self._region_exit_rewards(r, policy, starts)

            for a_abs in range(len(actions[r]), max_actions):
                P[a_abs, r, r] = 1.0
                R[a_abs, r, r] = -self.kappa

        return {
            "actions": actions,
            "transition": P,
            "reward": R,
            "max_actions": max_actions,
        }

    def _region_exit_probabilities(self, region, policy, start_states):
        states = self.regions[region]
        idx = self.local_index[region]
        region_set = self.region_sets[region]
        n = len(states)

        A = np.eye(n)
        B = np.zeros((n, self.n_regions))

        for i, s in enumerate(states):
            a = int(policy[s])
            row = self.model.transition_matrix[a].getrow(s)

            for s2, p in zip(row.indices, row.data):
                s2 = int(s2)
                if s2 in region_set:
                    A[i, idx[s2]] -= p
                else:
                    B[i, self.state_to_region[s2]] += p

        H = self._solve_linear(A, B)
        return H[[idx[s] for s in start_states]].mean(axis=0)

    def _region_exit_rewards(self, region, policy, start_states):
        """
        Reward analogue of Dean-Lin's destination-dependent c'_{RS}(pi).

        For each destination S:
            theta_i = sum_{j in S cap Periphery(R)} p_ij r_i
                      + sum_{j in R} p_ij (r_i + theta_j)
        """
        states = self.regions[region]
        idx = self.local_index[region]
        region_set = self.region_sets[region]
        n = len(states)
        rewards = np.zeros((n, self.n_regions))

        for target in self.neighbors[region]:
            if target == region:
                continue

            A = np.eye(n)
            b = np.zeros(n)

            for i, s in enumerate(states):
                a = int(policy[s])
                r_sa = float(self.model.reward_matrix[s, a])
                row = self.model.transition_matrix[a].getrow(s)

                for s2, p in zip(row.indices, row.data):
                    s2 = int(s2)
                    if s2 in region_set:
                        A[i, idx[s2]] -= p
                        b[i] += p * r_sa
                    elif self.state_to_region[s2] == target:
                        b[i] += p * r_sa

            rewards[:, target] = self._solve_linear(A, b)

        start_idx = [idx[s] for s in start_states]
        return rewards[start_idx].mean(axis=0)

    @staticmethod
    def _solve_linear(A, b):
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, b, rcond=None)[0]

    def _solve_abstract_mdp(self, abstract_model):
        P = abstract_model["transition"]
        R = abstract_model["reward"]
        actions = abstract_model["actions"]
        max_actions = abstract_model["max_actions"]

        v = np.zeros(self.n_regions)
        policy = np.zeros(self.n_regions, dtype=int)

        for _ in range(self.max_abstract_steps):
            old_v = v.copy()

            for r in range(self.n_regions):
                q = np.full(max_actions, -np.inf)
                for a_abs in range(len(actions[r])):
                    q[a_abs] = np.sum(R[a_abs, r] + self.discount * P[a_abs, r] * old_v)

                policy[r] = int(np.argmax(q))
                v[r] = q[policy[r]]

            self.steps += 1
            if np.max(np.abs(v - old_v)) < self.abstract_precision:
                break

        return v, policy

    def _glue_policy(self, local_policies):
        global_policy = np.zeros(self.model.state_dim, dtype=int)

        for r in range(self.n_regions):
            target = sorted(self.neighbors[r])[self.abstract_policy[r]]
            local_policy = local_policies[(r, target)]
            global_policy[self.regions[r]] = local_policy[self.regions[r]]

        return global_policy

    def _evaluate_global_policy(self, policy):
        n = self.model.state_dim
        P = np.zeros((n, n))
        r = np.zeros(n)

        for s in range(n):
            a = int(policy[s])
            r[s] = self.model.reward_matrix[s, a]
            row = self.model.transition_matrix[a].getrow(s)
            P[s, row.indices] = row.data

        return self._solve_linear(np.eye(n) - self.discount * P, r)
