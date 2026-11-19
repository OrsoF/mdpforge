"""Labeled Real-Time Dynamic Programming for positive-cost SSPs.

Bonet & Geffner (2003). Optimistic reward values (zero) correspond to
admissible lower cost heuristics; trials greedily sample successors and
check whether the greedy envelope is locally solved. Unlike start-state-only
LRTDP, seeds span every nonterminal state to match MDPForge's full-state
results. A separate global upper/lower gap check prevents a false gamma=1
convergence claim from small Bellman residuals.
"""

from __future__ import annotations

import numpy as np

from mdpforge.solvers.total._ssp_topology import SSPBase, validate_extra

METADATA = {
    "display_name": "SSP LRTDP · mdpforge",
    "family": "vi",
    "backend": "mdpforge",
    "description": "Labeled greedy sampled trials for positive-cost SSPs; global policy-gap certificate.",
}


class Solver(SSPBase):
    def __init__(
        self,
        model,
        final_precision=1e-3,
        *,
        seed=0,
        max_trial_steps=200,
        certify_every=10,
        max_iterations=100_000,
    ):
        super().__init__(model, final_precision, max_iterations=max_iterations)
        validate_extra("seed", seed, minimum=0)
        validate_extra("max_trial_steps", max_trial_steps)
        validate_extra("certify_every", certify_every)
        self.seed = seed
        self.max_trial_steps = max_trial_steps
        self.certify_every = certify_every
        self.n_trials = 0
        self.n_labeled = 0

    def run(self):
        p = self.problem
        upper = p.upper.copy()
        solved = np.zeros(p.n, dtype=bool)
        solved[list(p.goals)] = True
        rng = np.random.default_rng(self.seed)
        roots = list(map(int, p.transient))
        if not roots:
            result = p.certify(upper, iterations=0)
            return self.finish(result)

        def one_backup(s):
            upper[s] = min(upper[s], p.backup(s, upper))
            p.n_backups += 1
            return p.greedy(upper)[s]

        def check_solved(start):
            if solved[start]:
                return
            stack = [start]
            seen = set()
            stable = True
            while stack:
                s = stack.pop()
                if s in seen or solved[s]:
                    continue
                seen.add(s)
                proposed = p.backup(s, upper)
                if upper[s] - proposed > min(self.final_precision * 0.1, 1e-6):
                    stable = False
                else:
                    a = p.greedy(upper)[s]
                    stack.extend(int(t) for t in p.succ[s][a][0] if int(t) not in seen)
            if stable:
                solved[list(seen)] = True
            else:
                for s in reversed(sorted(seen)):
                    one_backup(s)

        for trial in range(1, self.max_iterations + 1):
            # Full-vector benchmark: cycle initial-state seeds through S.
            root = roots[(trial - 1) % len(roots)]
            if not solved[root]:
                trajectory = []
                state = root
                for _ in range(self.max_trial_steps):
                    if solved[state]:
                        break
                    action = one_backup(state)
                    trajectory.append(state)
                    successors, probabilities = p.succ[state][action]
                    state = int(rng.choice(successors, p=probabilities))
                for s in reversed(trajectory):
                    if not solved[s]:
                        check_solved(s)
            self.n_trials = trial
            if trial % self.certify_every == 0 or solved.all():
                result = p.certify(upper, iterations=trial)
                if result is not None:
                    self.n_labeled = int(solved.sum())
                    return self.finish(result)
                # Local labels are only search hints; cannot bypass global SSP bounds.
                if solved.all():
                    solved[:] = False
                    solved[list(p.goals)] = True
        raise RuntimeError(
            "SSP LRTDP: max_iterations reached without a global certificate"
        )
