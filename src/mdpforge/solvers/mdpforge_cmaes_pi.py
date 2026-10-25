"""Experimental policy iteration with CMA-ES policy evaluation."""

import numpy as np
from cma import CMAEvolutionStrategy

from mdpforge.core.mdp import MDP
from mdpforge.core.operators import bellman_operator, compute_transition_reward_policy
from mdpforge.core.solve_result import SolveResult
from mdpforge.core.solver import GenericSolver
from mdpforge.utils.solver_options import initial_vectors, solver_identifier

METADATA = {
    "display_name": "MPI · CMA-ES evaluation (experimental)",
    "family": "experimental",
    "backend": "cma",
    "description": (
        "Experimental policy iteration with CMA-ES minimization of the squared "
        "policy Bellman residual and independent final residual certification."
    ),
}


class Solver(GenericSolver):
    """Evaluate policies with CMA-ES and improve them greedily."""

    def __init__(
        self,
        model: MDP,
        discount: float,
        final_precision: float = 1e-3,
        *,
        initial_policy: np.ndarray | None = None,
        initial_value: np.ndarray | None = None,
        sigma0: float = 1.0,
        max_generations: int = 2000,
        max_iterations: int = 100,
        seed: int | None = None,
    ):
        """Configure this solver.

        final_precision: finite float > 0; absolute value error target; default 1e-3.
        initial_policy: None or an integer vector (S,) in [0, A); starting policy;
            default None.
        initial_value: None or a finite vector of shape (S,); starting value; default
            None.
        sigma0: finite float > 0; initial CMA-ES standard deviation; default 1.0.
        max_generations: int > 0; CMA-ES generations per policy; default 2000.
        max_iterations: int > 0; outer iteration budget; default 100.
        seed: None for the NumPy stream, or an int in [0, 2**32); RNG seed; default
            None.
        """
        super().__init__(
            model, discount, final_precision, max_iterations=max_iterations
        )

        value, policy = initial_vectors(model, initial_value, initial_policy)
        self.policy = np.zeros(model.state_dim, dtype=int) if policy is None else policy
        self.value = np.zeros(model.state_dim) if value is None else value
        self.sigma0 = float(sigma0)
        self.max_generations = max_generations
        # Derive the private RNG from the benchmark's seeded NumPy stream.
        seed = (
            int(np.random.randint(0, 2**32 - 1, dtype=np.int64))
            if seed is None
            else seed
        )
        self.rng = np.random.RandomState(seed)
        self.policy_iterations = 0
        self.generations = 0
        self.objective_evaluations = 0
        self.name = "mdpforge MPI (CMA-ES evaluation, experimental)"

        self.solver_id = solver_identifier(
            __name__,
            discount=self.discount,
            final_precision=self.final_precision,
            initial_policy=self.policy,
            initial_value=self.value,
            sigma0=self.sigma0,
            max_generations=self.max_generations,
            max_iterations=self.max_iterations,
            seed=seed,
        )

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

    def _solve(self) -> SolveResult:
        """Return only when both value and policy quality are certified."""
        tolerance = self.final_precision * (1 - self.discount)
        states = np.arange(self.model.state_dim)
        for _ in range(self.max_iterations):
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
                return SolveResult(np.asarray(self.value), None)
            self.policy = q_value.argmax(axis=1)

        raise RuntimeError(
            "CMA-ES policy iteration precision not reached: budget exhausted."
        )
