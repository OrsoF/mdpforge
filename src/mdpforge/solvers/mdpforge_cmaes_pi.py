"""Experimental policy iteration with CMA-ES policy evaluation."""

import numpy as np
from cma import CMAEvolutionStrategy

from mdpforge.core.model import MDPProtocol
from mdpforge.core.operators import bellman_operator, compute_transition_reward_policy
from mdpforge.core.solver import GenericSolver

METADATA = {
    "display_name": "MPI · CMA-ES evaluation (experimental)",
    "family": "mpi",
    "backend": "cma",
    "description": (
        "Experimental policy iteration with CMA-ES minimization of the squared "
        "policy Bellman residual and independent final residual certification."
    ),
    "update_scheme": "mixed",
}


class Solver(GenericSolver):
    """Evaluate policies with CMA-ES and improve them greedily."""

    def __init__(
        self,
        model: MDPProtocol,
        discount: float,
        final_precision: float = 1e-3,
        *,
        initial_policy: np.ndarray | None = None,
        initial_value: np.ndarray | None = None,
        sigma0: float = 1.0,
        max_generations: int = 2000,
        max_policy_iterations: int = 100,
        random_seed: int | None = None,
    ):
        super().__init__(model, discount, final_precision)
        if model.state_dim < 2:
            raise ValueError("CMA-ES evaluation requires at least two states.")
        if not np.isfinite(sigma0) or sigma0 <= 0:
            raise ValueError("sigma0 must be finite and positive.")
        for name, budget in (
            ("max_generations", max_generations),
            ("max_policy_iterations", max_policy_iterations),
        ):
            if type(budget) is not int or budget < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if random_seed is not None and (
            type(random_seed) is not int or not 0 <= random_seed < 2**32
        ):
            raise ValueError("random_seed must be None or an integer in [0, 2**32).")

        policy = (
            np.zeros(model.state_dim, dtype=int)
            if initial_policy is None
            else np.asarray(initial_policy)
        )
        if (
            policy.shape != (model.state_dim,)
            or not np.issubdtype(policy.dtype, np.integer)
            or np.any((policy < 0) | (policy >= model.action_dim))
        ):
            raise ValueError("initial_policy must contain one valid integer per state.")
        value = (
            np.zeros(model.state_dim)
            if initial_value is None
            else np.asarray(initial_value, dtype=float)
        )
        if value.shape != (model.state_dim,) or not np.all(np.isfinite(value)):
            raise ValueError("initial_value must contain one finite value per state.")

        self.policy = policy.astype(int, copy=True)
        self.value = value.copy()
        self.sigma0 = float(sigma0)
        self.max_generations = max_generations
        self.max_policy_iterations = max_policy_iterations
        # Derive the private RNG from the benchmark's seeded NumPy stream.
        seed = (
            int(np.random.randint(0, 2**32 - 1, dtype=np.int64))
            if random_seed is None
            else random_seed
        )
        self.rng = np.random.RandomState(seed)
        self.policy_iterations = 0
        self.generations = 0
        self.objective_evaluations = 0
        self.name = "mdpforge MPI (CMA-ES evaluation, experimental)"

    def _evaluate_policy(self, tolerance: float) -> np.ndarray:
        """Minimize the policy residual quadratic to a certified inner tolerance."""
        transition, reward = compute_transition_reward_policy(self.model, self.policy)

        def residual(value):
            return value - self.discount * transition.dot(value) - reward

        best_value = self.value.copy()
        best_residual = residual(best_value)
        best_fitness = 0.5 * np.dot(best_residual, best_residual)
        if np.max(np.abs(best_residual)) <= tolerance:
            return best_value

        optimizer = CMAEvolutionStrategy(
            best_value,
            self.sigma0,
            {
                "randn": self.rng.randn,
                "seed": np.nan,
                "verbose": -9,
                "verb_log": 0,
                "signals_filename": "",
                "maxiter": self.max_generations,
                # Objective differences are not a value-error certificate.
                "tolfun": 0,
                "tolfunhist": 0,
                "tolx": 0,
            },
        )
        for _ in range(self.max_generations):
            candidates = optimizer.ask()
            fitness = []
            for candidate in candidates:
                candidate_residual = residual(candidate)
                score = 0.5 * np.dot(candidate_residual, candidate_residual)
                self.objective_evaluations += 1
                if not np.isfinite(score):
                    raise RuntimeError("CMA-ES produced a nonfinite policy objective.")
                fitness.append(float(score))
                if score < best_fitness:
                    best_fitness = score
                    best_value = np.asarray(candidate).copy()
                    best_residual = candidate_residual
            optimizer.tell(candidates, fitness)
            self.generations += 1
            # The distribution mean can outperform every sampled candidate.
            mean = np.asarray(optimizer.mean)
            mean_residual = residual(mean)
            mean_fitness = 0.5 * np.dot(mean_residual, mean_residual)
            self.objective_evaluations += 1
            if not np.isfinite(mean_fitness):
                raise RuntimeError("CMA-ES produced a nonfinite policy objective.")
            if mean_fitness < best_fitness:
                best_fitness = mean_fitness
                best_value = mean.copy()
                best_residual = mean_residual
            if np.max(np.abs(best_residual)) <= tolerance:
                return best_value
            if optimizer.stop():
                break

        error_bound = np.max(np.abs(best_residual)) / (1 - self.discount)
        raise RuntimeError(
            "CMA-ES policy evaluation precision not reached: "
            f"error bound {error_bound:g}; budget or optimizer stopping criterion."
        )

    def _solve(self) -> np.ndarray:
        """Return only when both value and policy quality are certified."""
        tolerance = self.final_precision * (1 - self.discount)
        states = np.arange(self.model.state_dim)
        for _ in range(self.max_policy_iterations):
            self.value = self._evaluate_policy(tolerance / 4)
            self.policy_iterations += 1
            q_value = bellman_operator(self.model, self.value, self.discount)
            optimal_residual = np.max(np.abs(q_value.max(axis=1) - self.value))
            policy_residual = np.max(np.abs(q_value[states, self.policy] - self.value))
            # Triangle inequality also certifies the returned policy's value.
            if (
                np.isfinite(optimal_residual + policy_residual)
                and optimal_residual + policy_residual <= tolerance
            ):
                return self.value
            self.policy = q_value.argmax(axis=1)

        raise RuntimeError(
            "CMA-ES policy iteration precision not reached: budget exhausted."
        )
