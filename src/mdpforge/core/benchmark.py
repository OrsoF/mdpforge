"""Benchmark finite discounted MDP solvers at a fixed value precision."""

import csv
import random
import warnings
from copy import deepcopy
from importlib import import_module
from pathlib import Path
from pkgutil import iter_modules
from time import perf_counter

import numpy as np

from mdpforge import models, solvers
from mdpforge.core.mdp import MDP
from mdpforge.core.model import MDPProtocol
from mdpforge.core.operators import optimal_bellman_operator
from mdpforge.core.validation import validate_model


def _catalogue(package):
    """List public catalogue modules without importing their dependencies."""
    return {
        module.name
        for module in iter_modules(package.__path__)
        if not module.ispkg and not module.name.startswith("_")
    }


class Benchmark:
    """Compare catalogue, matrix-defined or already-built finite discounted MDPs.

    VI is registered by default; use default_solvers=False to select
    only your own solvers.

    Models must expose ``name``, ``state_dim``, ``action_dim``,
    ``transition_matrix`` and ``reward_matrix``. Solver functions accept
    ``(transitions, rewards, discount, precision)`` and return a value vector.
    Solver classes accept ``(model, discount, final_precision=...)`` and provide
    ``run()`` and a ``value`` vector.

    Runtime includes function execution or class construction and ``run()``,
    but excludes model copying, validation and independent precision verification.
    """

    def __init__(self, *, default_solvers=True):
        self._mdps: list[MDPProtocol] = []
        self._solvers = {}  # label -> (solver class/function, options)
        self.results = []
        if default_solvers:
            from mdpforge.solvers.personal_vi import Solver as VI

            self.add_solver("VI", solve_function=VI)

    def add_mdp(self, model: MDPProtocol | str, *, transitions=None, reward=None):
        """Register a catalogue name, custom matrices or a built MDP.

        Catalogue models use Model() and create_model() with their defaults.
        Unknown names require transitions and reward, converted to
        CSR and validated with dimensions inferred from rewards of shape (S, A).
        Each built MDP name must be unique. Construction is outside solver timing.
        """
        if isinstance(model, str):
            if not model.strip():
                raise ValueError("Each MDP must have a nonempty string 'name'.")
            catalogue = _catalogue(models)
            if model in catalogue:
                if transitions is not None or reward is not None:
                    raise ValueError(
                        f"Catalogue model {model!r} uses its generator; "
                        "choose a custom name when supplying matrices."
                    )
                model = import_module(f"{models.__name__}.{model}").Model()
                model.create_model()
            else:
                if transitions is None or reward is None:
                    raise ValueError(
                        f"Unknown model {model!r}. Provide both transitions and reward "
                        "for a custom MDP. Available models: "
                        + ", ".join(sorted(catalogue))
                    )
                model = MDP.from_matrices(model, transitions, reward)
        elif transitions is not None or reward is not None:
            raise ValueError("Pass a name when providing transitions and reward.")
        name = getattr(model, "name", None)
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Each MDP must have a nonempty string 'name'.")
        if any(existing.name == name for existing in self._mdps):
            raise ValueError(f"An MDP named {name!r} is already registered.")
        self._mdps.append(model)
        self.results = []
        return self

    def add_all_models(self):
        """Add all catalogue models with default dimensions, without duplicates.

        Missing dependencies are skipped with a warning; other errors propagate.
        """
        for name in sorted(_catalogue(models)):
            try:
                model = import_module(f"{models.__name__}.{name}").Model()
                if any(existing.name == model.name for existing in self._mdps):
                    continue
                model.create_model()
                self.add_mdp(model)
            except ModuleNotFoundError as exc:
                warnings.warn(
                    f"Skipping model {name!r}: missing dependency {exc.name!r}",
                    stacklevel=2,
                )
        return self

    def add_solver(self, solver_name: str, *, solve_function=None, **options):
        """Register a catalogue solver or a named custom function/class.

        Without solve_function, solver_name identifies a module in solvers/.
        Otherwise it labels the supplied callable. Functions receive
        (CSR transitions, NumPy rewards, discount, precision) and return values.
        Classes receive (model, discount, final_precision=...) and expose run()
        and value. Options are forwarded; problem data, discount and precision
        are supplied by run() and cannot be overridden per solver.
        """
        if not isinstance(solver_name, str) or not solver_name.strip():
            raise ValueError("A solver must have a nonempty string name.")
        if solver_name in self._solvers:
            raise ValueError(f"A solver named {solver_name!r} is already registered.")
        reserved = {
            "model",
            "transitions",
            "rewards",
            "discount",
            "precision",
            "final_precision",
        } & options.keys()
        if reserved:
            raise ValueError(
                f"Solver parameters controlled by run(): {sorted(reserved)}"
            )
        if solve_function is None:
            catalogue = _catalogue(solvers)
            if solver_name not in catalogue:
                raise ValueError(
                    f"Unknown solver {solver_name!r}. Available solvers: "
                    + ", ".join(sorted(catalogue))
                )
            solve_function = import_module(f"{solvers.__name__}.{solver_name}").Solver
        if not callable(solve_function):
            raise TypeError("solve_function must be a callable function or class.")
        self._solvers[solver_name] = (solve_function, options)
        self.results = []
        return self

    def add_all_solvers(self):
        """Add all catalogue solvers with defaults, preserving existing entries.

        Default VI is not duplicated under its catalogue name. Missing
        dependencies are skipped with a warning; other errors propagate.
        """
        for name in sorted(_catalogue(solvers)):
            if name in self._solvers:
                continue
            try:
                constructor = import_module(f"{solvers.__name__}.{name}").Solver
            except ModuleNotFoundError as exc:
                warnings.warn(
                    f"Skipping solver {name!r}: missing dependency {exc.name!r}",
                    stacklevel=2,
                )
                continue
            if any(
                existing is constructor and not options
                for existing, options in self._solvers.values()
            ):
                continue
            self.add_solver(name, solve_function=constructor)
        return self

    def run(self, discount, precision=1e-3, *, repeats=3, seed=0, verbose=False):
        """Run every registered MDP/solver pair at a common value precision.

        ``precision`` is an upper bound on ||V - V*||_infinity, certified by
        ||T V - V||_infinity / (1 - discount). All failures are recorded rather
        than interrupting the remaining comparisons.

        ``verbose=True`` prints trial progress, runtime and failure details
        outside solver timing.

        Each solver receives an independent copy of its MDP. Seeds are paired
        across solvers and MDPs for each repeat. Global NumPy and Python RNG
        states are restored; solvers using private RNGs must seed those
        generators themselves.
        """
        if not np.isfinite(discount) or not 0 < discount < 1:
            raise ValueError("discount must be strictly between 0 and 1")
        if not np.isfinite(precision) or precision <= 0:
            raise ValueError("precision must be finite and positive")
        if type(repeats) is not int or repeats < 1:
            raise ValueError("repeats must be a positive integer")
        if type(seed) is not int or not 0 <= seed < 2**32 - repeats:
            raise ValueError("seed must allow valid NumPy seeds for every repeat")
        if not self._mdps:
            raise ValueError("No MDP registered; call add_mdp() first")
        if not self._solvers:
            raise ValueError("No solver registered; call add_solver() first")

        for model in self._mdps:
            validate_model(model)

        self.results = []
        numpy_state, random_state = np.random.get_state(), random.getstate()
        try:
            for model in self._mdps:
                for solver_name, (constructor, kwargs) in self._solvers.items():
                    for repeat in range(repeats):
                        if verbose:
                            print(
                                f"{model.name} / {solver_name} "
                                f"[{repeat + 1}/{repeats}]: ",
                                end="",
                                flush=True,
                            )
                        runtime = residual = error_bound = np.nan
                        status, error = "error", ""
                        start = None
                        try:
                            trial_model = deepcopy(model)
                            np.random.seed(seed + repeat)
                            random.seed(seed + repeat)
                            start = perf_counter()
                            if isinstance(constructor, type):
                                solver = constructor(
                                    trial_model,
                                    discount,
                                    final_precision=precision,
                                    **kwargs,
                                )
                                solver.run()
                                value = solver.value
                            else:
                                value = constructor(
                                    trial_model.transition_matrix,
                                    trial_model.reward_matrix,
                                    discount,
                                    precision,
                                    **kwargs,
                                )
                            runtime = perf_counter() - start

                            value = np.asarray(value)
                            if value.shape != (model.state_dim,) or not np.all(
                                np.isfinite(value)
                            ):
                                raise ValueError(
                                    "Solver values must be a finite vector of state_dim"
                                )

                            residual = float(
                                np.max(
                                    np.abs(
                                        optimal_bellman_operator(model, value, discount)
                                        - value
                                    )
                                )
                            )
                            error_bound = residual / (1 - discount)
                            status = (
                                "success" if error_bound <= precision else "imprecise"
                            )
                            if status == "imprecise":
                                error = (
                                    f"Value error bound {error_bound:g} "
                                    f"exceeds precision {precision:g}"
                                )
                        except Exception as exc:
                            if start is not None and np.isnan(runtime):
                                runtime = perf_counter() - start
                            error = f"{type(exc).__name__}: {exc}"
                        self.results.append(
                            {
                                "model": model.name,
                                "state_dim": model.state_dim,
                                "action_dim": model.action_dim,
                                "solver": solver_name,
                                "repeat": repeat + 1,
                                "seed": seed + repeat,
                                "discount": discount,
                                "epsilon": precision,  # Keep the original CSV schema.
                                "runtime": runtime,
                                "residual": residual,
                                "error_bound": error_bound,
                                "status": status,
                                "error": error,
                            }
                        )
                        if verbose:
                            print(
                                f"{status} ({runtime:.3f} s)"
                                + (f" — {error}" if error else ""),
                                flush=True,
                            )
        finally:
            np.random.set_state(numpy_state)
            random.setstate(random_state)
        return self.results

    def plot_heat(self, *, reference=None, metric="speedup", ax=None, show=True):
        """Plot a solver-by-MDP heatmap from the last run.

        ``metric='speedup'`` shows median reference runtime / median solver
        runtime; ``metric='runtime'`` shows median runtime in seconds.
        A cell is only plotted when *all* repeats met the target precision.
        Failed/imprecise trials are marked FAIL, not silently discarded.

        Returns ``(figure, axes)``. Matplotlib is an optional dependency.
        """
        from mdpforge.utils.plotting import plot_heat

        return plot_heat(
            self.results,
            [model.name for model in self._mdps],
            list(self._solvers),
            reference=reference,
            metric=metric,
            ax=ax,
            show=show,
        )

    def export_csv(self, path):
        """Export the latest run without rerunning any solver."""
        if not self.results:
            raise ValueError("No benchmark measurements to export")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.results[0].keys())
            writer.writeheader()
            writer.writerows(self.results)
        return path
