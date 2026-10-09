"""Finite-state approximation of the impatient-queue MDP.

Based on Hyon & Jean-Marie (2012), *Scheduling Services in a Queuing
System with Impatience and Setup Costs*, The Computer Journal 55(5).

State y: number of customers waiting, 0 <= y <= N.
Action q: number of customers admitted to service, 0 <= q < action_dim.

The original article uses q in {0, 1} and an unbounded queue. Here larger
batches are allowed and the next state is capped at N:

    X_next = min(N, Binomial(max(0, y - q), p_stay) + Poisson(arrival_rate)).

The probability mass above capacity N is assigned to state N (not lost).
Rewards are the *negative* of the article's expected one-step costs.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.stats import poisson

from mdpforge.core.mdp import MDP


def _build_base_kernel(N: int, p: float, lam: float, eps: float) -> csr_matrix:
    """Build sparse P[n, z] = Pr(X_next=z | n customers after service).

    The recurrence is the convolution with Bernoulli(p): P[n+1] is
    obtained from P[n] with one shift and weighted addition. Small tails
    are folded into their nearest retained endpoint, giving stochastic
    rows rather than adding missing probability to an arbitrary state.

    Total variation error per row is at most eps (up to floating-point
    rounding): each update moves at most 2 * eps / (2 * (N+1)) mass,
    and Markov kernels contract total variation distance.
    """
    tail_tol = eps / (2 * (N + 1))

    # Distribution for n=0: capped Poisson, initially trimmed via quantiles.
    lo = min(N, max(0, int(poisson.ppf(tail_tol, lam))))
    hi = min(N, max(lo, int(poisson.isf(tail_tol, lam))))

    if lo == hi:
        probs = np.array([1.0])
    else:
        probs = poisson.pmf(np.arange(lo, hi + 1), lam)
        probs[0] = poisson.cdf(lo, lam)  # Move omitted lower tail to lo.
        probs[-1] = poisson.sf(hi - 1, lam)  # Include full upper tail at hi.

    data = []
    indices = []
    indptr = np.empty(N + 2, dtype=np.int64)
    indptr[0] = 0

    for n in range(N + 1):
        data.append(probs)
        indices.append(np.arange(lo, lo + len(probs), dtype=np.int32))
        indptr[n + 1] = indptr[n] + len(probs)

        if n == N:
            break

        # Convolve current row with Bernoulli(p) in linear time in its width.
        nxt = np.empty(len(probs) + 1)
        nxt[0] = (1 - p) * probs[0]
        nxt[-1] = p * probs[-1]
        nxt[1:-1] = (1 - p) * probs[1:] + p * probs[:-1]

        # Preserve the exact finite-capacity boundary during the update.
        if lo + len(nxt) - 1 > N:
            nxt[-2] += nxt[-1]
            nxt = nxt[:-1]

        # Trim at most tail_tol mass on each side, folding it into the
        # nearest retained state. No artificial self-transition is added.
        left = int(np.searchsorted(np.cumsum(nxt), tail_tol, side="right"))
        right = int(np.searchsorted(np.cumsum(nxt[::-1]), tail_tol, side="right"))
        end = len(nxt) - right

        probs = nxt[left:end].copy()
        if left:
            probs[0] += nxt[:left].sum()
        if right:
            probs[-1] += nxt[end:].sum()
        lo += left

    return csr_matrix(
        (np.concatenate(data), np.concatenate(indices), indptr),
        shape=(N + 1, N + 1),
    )


class Model(MDP):
    """Impatient-queue MDP with a sparse, accuracy-controlled kernel.

    Parameters
    ----------
    state_dim : int
        Number of states; queue capacity is state_dim - 1.
    action_dim : int
        Number of actions, interpreted as service batch sizes 0, 1, ...
    truncation_tol : float
        Upper bound on total-variation error for any base transition row.
    """

    def __init__(
        self,
        state_dim: int = 200,
        action_dim: int = 10,
        *,
        truncation_tol: float = 1e-8,
    ):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        if self.state_dim < 1 or self.action_dim < 1:
            raise ValueError("state_dim and action_dim must be positive")
        if not np.isfinite(truncation_tol) or not 0 < truncation_tol < 1:
            raise ValueError("truncation_tol must be in (0, 1)")

        self.max_queue_length = self.state_dim - 1
        self.nb_serv = self.action_dim - 1
        self.proba_person_stay = 0.9
        self.arrival_poisson_rate = 0.9
        self.batch_cost = 1.5
        self.loss_cost = 2.0
        self.holding_cost = 0.1
        self.q_cost = (1 - self.proba_person_stay) * self.loss_cost + self.holding_cost

        self.truncation_tol = float(truncation_tol)
        self.name = f"{self.state_dim}_{self.action_dim}_queue_impatience"

    def _build_model(self):
        N = self.max_queue_length
        p = self.proba_person_stay
        lam = self.arrival_poisson_rate
        if not 0 <= p <= 1 or not np.isfinite(p):
            raise ValueError("proba_person_stay must be in [0, 1]")
        if lam < 0 or not np.isfinite(lam):
            raise ValueError("arrival_poisson_rate must be finite and nonnegative")

        states = np.arange(self.state_dim)
        actions = np.arange(self.action_dim)
        remaining = np.maximum(0, states[:, None] - actions[None, :])

        # mdpforge algorithms maximize reward; the paper minimizes cost.
        self.q_cost = (1 - p) * self.loss_cost + self.holding_cost
        self.reward_matrix = -(
            self.batch_cost * actions[None, :] + self.q_cost * remaining
        )

        # There are only N+1 distinct transition rows across *all* actions.
        base_transition_matrix = _build_base_kernel(N, p, lam, self.truncation_tol)

        self.transition_matrix = [
            base_transition_matrix[np.maximum(0, states - q)] for q in actions
        ]
