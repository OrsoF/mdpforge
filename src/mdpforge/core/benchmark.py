"""Benchmark finite discounted MDP solvers at a fixed value precision."""

import random
from collections.abc import Sequence
from time import perf_counter

import numpy as np

from mdpforge import models
from mdpforge.core.benchmark_result import BenchmarkResult
from mdpforge.core.mdp import MDP
from mdpforge.core.validation import validate_result
from mdpforge.solvers import is_catalogue_solver, list_solvers
from mdpforge.utils.catalogue import (
    _available_solvers,
    _load_model,
    _load_solver,
    _select_models,
)
from mdpforge.utils.experiment import (
    _check_solver_options,
    _isolated_solver_trial,
    _solver_trial,
    capture_experiment,
)


def _same_options(left, right):
    """Compare explicit options conservatively, including initial-value arrays."""
    if left.keys() != right.keys():
        return False
    for key, value in left.items():
        other = right[key]
        if value is other:
            continue
        try:
            if isinstance(value, np.ndarray) and isinstance(other, np.ndarray):
                equal = value.dtype == other.dtype and np.array_equal(
                    value, other, equal_nan=True
                )
            else:
                equal = value == other
            if not isinstance(equal, (bool, np.bool_)) or not equal:
                return False
        except (TypeError, ValueError):
            return False
    return True


class Benchmark:
    """Compare catalogue, matrix-defined or already-built finite discounted MDPs.

    Models must expose ``name``, ``state_dim``, ``action_dim``,
    ``transition_matrix`` and ``reward_matrix``. Solver functions accept
    ``(transitions, rewards, discount, precision)`` and return a value vector.
    Solver classes accept ``(model, discount)`` with optional ``final_precision``
    and return a ``SolveResult`` from ``run()``.

    Runtime includes function execution or class construction and ``run()``,
    but excludes model copying and experiment bookkeeping.
    """

    def __init__(self):
        self._mdps: list[MDP] = []
        self._solvers = {}  # unique key -> (solver class/function, options)

    def reset(self, *, models=True, solvers=True):
        """Reset registrations; previously returned BenchmarkResult objects are kept.

        models: True removes all registered MDPs; False keeps them.
        solvers: True removes all solvers; False keeps them and their options.

        Return self for chaining. Cached models and exported files are kept.
        """
        if models:
            self._mdps.clear()
        if solvers:
            self._solvers.clear()
        return self

    def add_mdp(
        self,
        model: MDP | str | Sequence[str] | None = None,
        *,
        transitions=None,
        reward=None,
        size: str | None = None,
        category: str | None = None,
        verbose=False,
    ):
        """Add MDPs by name, matrices, objects, or metadata filters.

        Args:
            model: Catalogue ID(s), built MDP, or custom name.
            transitions: Transition matrices (S, S), one per action.
            reward: Reward matrix (S, A).
            size: "small", "medium", or "large".
            category: Model category (e.g., "navigation", "abstract").
            verbose: Whether to print loading progress.

        Returns:
            self: Current instance for chaining.

        Notes:
            Filters use AND and apply only to catalogue models.
            Custom names must be unique. Groups skip unavailable models
            and duplicates. Explicit IDs must match filters.
            Use `add_all_mdps()` to select the entire catalogue.
        """
        named_group = isinstance(model, Sequence) and not isinstance(model, str)
        if named_group and (
            not model
            or any(not isinstance(name, str) or not name.strip() for name in model)
        ):
            raise ValueError("model must be a nonempty sequence of catalogue names.")
        if named_group or size is not None or category is not None:
            if (
                (model is not None and not isinstance(model, str) and not named_group)
                or transitions is not None
                or reward is not None
            ):
                raise ValueError(
                    "Catalogue selections cannot be combined with built objects "
                    "or matrices."
                )
            if isinstance(model, str) and not model.strip():
                raise ValueError("Each MDP must have a nonempty string 'name'.")
            names = None
            if model is not None:
                names = list(dict.fromkeys(model)) if named_group else [model]
            for entry in _select_models(names, size=size, category=category):
                preset = entry["sizes"][size] if size is not None else None
                self._add_catalogue_model(
                    entry,
                    preset=preset,
                    grouped=named_group or model is None,
                    verbose=verbose,
                )
            return self
        if model is None:
            raise ValueError(
                "Provide a model or at least one size/category filter; "
                "use add_all_mdps() to add the entire catalogue."
            )
        if isinstance(model, str):
            if not model.strip():
                raise ValueError("Each MDP must have a nonempty string 'name'.")
            catalogue = {entry["name"]: entry for entry in models.list_models()}
            if model in catalogue:
                if transitions is not None or reward is not None:
                    raise ValueError(
                        f"Catalogue model {model!r} uses its generator; "
                        "choose a custom name when supplying matrices."
                    )
                return self._add_catalogue_model(catalogue[model], verbose=verbose)
            else:
                if transitions is None or reward is None:
                    raise ValueError(
                        f"Unknown model {model!r}. Provide both transitions and reward "
                        "for a custom MDP. Available models: "
                        + ", ".join(sorted(catalogue))
                    )
                if verbose:
                    print(f"Creating MDP {model!r} from matrices...", flush=True)
                model = MDP.from_matrices(model, transitions, reward)
        elif transitions is not None or reward is not None:
            raise ValueError("Pass a name when providing transitions and reward.")
        name = getattr(model, "name", None)
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Each MDP must have a nonempty string 'name'.")
        if any(existing.name == name for existing in self._mdps):
            raise ValueError(f"An MDP named {name!r} is already registered.")
        self._mdps.append(model)
        if verbose:
            print(f"Added MDP {name!r}", flush=True)
        return self

    def remove_model(self, model_name: str):
        """Remove a registered model by name and return self.

        model_name: exact registered name or unique catalogue ID.
        """
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a nonempty string")
        for index, model in enumerate(self._mdps):
            if model.name == model_name:
                del self._mdps[index]
                return self
        matches = [
            index
            for index, model in enumerate(self._mdps)
            if type(model).__module__ == f"{models.__name__}.{model_name}"
        ]
        if len(matches) == 1:
            del self._mdps[matches[0]]
            return self
        if matches:
            raise ValueError(
                f"Multiple models from {model_name!r} are registered; "
                "use the exact model.name."
            )
        raise ValueError(f"Model {model_name!r} is not registered.")

    def _add_catalogue_model(self, entry, *, preset=None, grouped=False, verbose=False):
        """Build a recipe; groups skip duplicates and missing dependencies."""
        name = entry["name"]
        if verbose:
            start = perf_counter()
            print(f"Loading/building model {name!r}...", flush=True)
        model = _load_model(
            entry,
            preset=preset,
            registered=self._mdps,
            grouped=grouped,
            verbose=verbose,
        )
        if model is not None:
            self.add_mdp(model)
            if verbose:
                model_label = getattr(model, "display_name", model.name)
                print(
                    f"Added MDP {model_label!r} "
                    f"({model.state_dim} states, {model.action_dim} actions) "
                    f"in {perf_counter() - start:.3f} s",
                    flush=True,
                )
        return self

    def add_all_mdps(self, *, size: str | None = "small", verbose=False):
        """Add all catalogue MDPs for an optional size preset; return self.

        size: "small", "medium" or "large"; None uses recipe defaults.
        verbose: True prints model-loading progress; False keeps additions quiet.
        Unavailable sizes, duplicates and missing dependencies are skipped.
        """
        for entry in models.list_models(size=size, available=True):
            preset = entry["sizes"][size] if size is not None else None
            self._add_catalogue_model(
                entry, preset=preset, grouped=True, verbose=verbose
            )
        return self

    def add_default_mdps(self, *, verbose=False):
        """Add small Rooms, Mountain Car and Garnet models; return self."""
        return self.add_mdp(
            ("rooms", "mountain", "garnet"), size="small", verbose=verbose
        )

    def add_solver(
        self,
        solver_name: str | None = None,
        *,
        solve_function=None,
        family=None,
        backend=None,
        **options,
    ):
        """Add solvers by name, metadata filters, or custom implementation.

        Args:
            solver_name: Solver ID from `list_solvers()` or custom name.
                If None, select solvers using metadata filters.
            solve_function: Custom solver function or class.
                Requires `solver_name` and excludes metadata filters.
            family: Algorithm family ("vi", "pi", "mpi", "lp", "experimental").
            backend: Implementation library (e.g., "mdpforge", "gurobi").
            **options: Solver-specific arguments passed at initialization.

        Returns:
            self: Current instance for method chaining.

        Notes:
            Filters accept strings, non-empty sequences, or None.
            Multiple filters use AND; values within a filter use OR.
        """
        filters = {
            "family": family,
            "backend": backend,
        }
        filtered = any(value is not None for value in filters.values())
        grouped = solver_name is None and filtered
        if not grouped and (
            not isinstance(solver_name, str) or not solver_name.strip()
        ):
            raise ValueError("A solver must have a nonempty string name.")
        if filtered and solve_function is not None:
            raise ValueError(
                "Metadata filters cannot accompany a custom solve_function"
            )
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
        if grouped:
            return self._add_solver_group(list_solvers(**filters), options)
        if solve_function is None:
            catalogue = {entry["name"]: entry for entry in list_solvers()}
            if solver_name not in catalogue:
                raise ValueError(
                    f"Unknown solver {solver_name!r}. Available solvers: "
                    + ", ".join(sorted(catalogue))
                )
            if filtered and solver_name not in {
                entry["name"] for entry in list_solvers(**filters)
            }:
                raise ValueError(
                    f"Solver {solver_name!r} does not match the requested "
                    "metadata filters"
                )
            solve_function = _load_solver(
                solver_name, backend=catalogue[solver_name]["backend"]
            )
        self._register_solver(solver_name, solve_function, options)
        return self

    def _register_solver(
        self, name, constructor, options, *, registry=None, skip_existing=False
    ):
        """Assign a unique key and register one solver configuration."""
        if not callable(constructor):
            raise TypeError("solve_function must be a callable function or class.")
        _check_solver_options(name, constructor, options)
        if registry is None:
            registry = self._solvers
        auto_name = (
            is_catalogue_solver(constructor)
            and name == constructor.__module__.rsplit(".", 1)[-1]
        )
        if auto_name or skip_existing:
            for registered_name, (existing, previous_options) in registry.items():
                if existing is constructor and _same_options(previous_options, options):
                    if skip_existing:
                        return
                    raise ValueError(
                        f"A solver named {registered_name!r} is already registered."
                    )
        if auto_name:
            base_name = name
            index = 2
            while name in registry:
                name = f"{base_name}_{index}"
                index += 1
        elif name in registry:
            raise ValueError(f"A solver named {name!r} is already registered.")
        registry[name] = (constructor, options.copy())

    def remove_solver(self, solver_name: str):
        """Remove a registered solver by label and return self.

        solver_name: registered ID, including the numeric suffix for a configuration.
        """
        if not isinstance(solver_name, str) or not solver_name.strip():
            raise ValueError("A solver must have a nonempty string name.")
        if solver_name not in self._solvers:
            raise ValueError(f"Solver {solver_name!r} is not registered.")
        del self._solvers[solver_name]
        return self

    def _add_solver_group(self, entries, options):
        """Check the entire selection before committing new registrations."""
        registry = self._solvers.copy()
        for name, constructor in _available_solvers(entries):
            self._register_solver(
                name, constructor, options, registry=registry, skip_existing=True
            )
        self._solvers.update(registry)
        return self

    def add_default_solvers(self):
        """Add catalogue mdpforge VI and MPI solvers; return self."""
        registry = self._solvers.copy()
        for name in ("mdpforge_vi", "mdpforge_mpi"):
            constructor = _load_solver(name)
            if any(existing is constructor for existing, _ in registry.values()):
                continue
            if name in registry:
                raise ValueError(f"Solver {name!r} has a different implementation")
            self._register_solver(name, constructor, {}, registry=registry)
        self._solvers.update(registry)
        return self

    def add_all_solvers(self):
        """Add all catalogue solvers, preserving existing entries."""
        entries = (
            entry for entry in list_solvers() if entry["name"] not in self._solvers
        )
        return self._add_solver_group(entries, {})

    def run(
        self,
        discount,
        precision=1e-3,
        *,
        repeats=3,
        seed=0,
        verbose=False,
        timeout=None,
        validate=False,
    ) -> BenchmarkResult:
        """Run each MDP/solver pair and return an independent BenchmarkResult.

        ``precision`` is passed to each solver. All failures are recorded rather
        than interrupting the remaining comparisons.

        ``validate``: False skips certification (default); True validates each
        result outside solver timing and releases it before the next trial.
        Validation status, errors and runtime are recorded separately;
        ``success`` always means execution completed, not certified accuracy.
        ``validation_status`` is ``not_run``, ``passed`` or ``failed``;
        ``error_bound`` is available only when certification passes.

        ``verbose=True`` prints trial progress, runtime and failure details
        outside solver timing.

        Each solver receives an independent copy of its MDP. Seeds are paired
        across solvers and MDPs for each repeat. Global NumPy and Python RNG
        states are restored; solvers using private RNGs must seed those
        generators themselves.

        ``timeout`` is an optional positive wall-clock limit in seconds per
        trial. It uses a fresh spawn process and covers child startup, model
        copying, solver construction/execution and result serialization, after
        dispatch from the parent. Parent input serialization is outside this
        limit. Process cleanup can take up to two
        extra seconds. Timed-out or crashed trials have no solver runtime (NaN).
        Validation runs in the parent and is outside the timeout.

        With a timeout, models, solvers and options must be pickleable; custom
        functions/classes must be importable from a module. In scripts, protect
        the benchmark call with ``if __name__ == "__main__":``. Only the dedicated
        worker is terminated, not any descendant processes started by a custom solver.

        The result's ``experiment`` captures settings, versions and
        machine information before solver timing. Each row includes its ID.
        """
        if not np.isfinite(discount) or not 0 < discount < 1:
            raise ValueError("discount must be strictly between 0 and 1")
        if not np.isfinite(precision) or precision <= 0:
            raise ValueError("precision must be finite and positive")
        if type(repeats) is not int or repeats < 1:
            raise ValueError("repeats must be a positive integer")
        if type(seed) is not int or not 0 <= seed < 2**32 - repeats:
            raise ValueError("seed must allow valid NumPy seeds for every repeat")
        if type(validate) is not bool:
            raise ValueError("validate must be a bool")
        if timeout is not None and (
            isinstance(timeout, (bool, np.bool_))
            or not isinstance(timeout, (int, float, np.integer, np.floating))
            or not np.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("timeout must be finite and positive, or None")
        if not self._mdps:
            raise ValueError("No MDP registered; call add_mdp() first")
        if not self._solvers:
            raise ValueError("No solver registered; call add_solver() first")

        rows = []
        numpy_state, random_state = np.random.get_state(), random.getstate()
        try:
            experiment = capture_experiment(
                self._mdps,
                self._solvers,
                discount,
                precision,
                repeats,
                seed,
                timeout=timeout,
                validate=validate,
            )
            for model in self._mdps:
                model_label = getattr(model, "display_name", model.name)
                for solver_name, (constructor, kwargs) in self._solvers.items():
                    for repeat in range(repeats):
                        if verbose:
                            print(
                                f"{model_label} / {solver_name} "
                                f"[{repeat + 1}/{repeats}]: ",
                                end="",
                                flush=True,
                            )
                        runtime = np.nan
                        result = None
                        validation_method = "unknown"
                        validation_status, validation_error = "not_run", ""
                        validation_runtime, error_bound = np.nan, np.nan
                        status, error = "error", ""
                        try:
                            args = (
                                model,
                                constructor,
                                kwargs,
                                discount,
                                precision,
                                seed + repeat,
                                solver_name,
                            )
                            if timeout is None:
                                result, runtime, status, error = _solver_trial(*args)
                            else:
                                result, runtime, status, error = _isolated_solver_trial(
                                    *args, timeout
                                )
                            if status == "success":
                                validation_method = getattr(
                                    result, "validation_method", "unknown"
                                )
                        except Exception as exc:
                            status = "error"
                            error = f"{type(exc).__name__}: {exc}"
                        if validate and status == "success":
                            validation_start = perf_counter()
                            try:
                                error_bound = validate_result(
                                    model, result, discount=discount, precision=precision
                                )
                                validation_status = "passed"
                            except Exception as exc:
                                validation_status = "failed"
                                validation_error = f"{type(exc).__name__}: {exc}"
                            finally:
                                validation_runtime = perf_counter() - validation_start
                        del result
                        rows.append(
                            {
                                "experiment_id": experiment["experiment_id"],
                                "model": model.name,
                                "state_dim": model.state_dim,
                                "requested_state_dim": getattr(
                                    model, "requested_state_dim", None
                                ),
                                "action_dim": model.action_dim,
                                "solver": solver_name,
                                "repeat": repeat + 1,
                                "seed": seed + repeat,
                                "discount": discount,
                                "epsilon": precision,  # Keep the original CSV schema.
                                "runtime": runtime,
                                "validation_method": validation_method,
                                "validation_status": validation_status,
                                "validation_error": validation_error,
                                "validation_runtime": validation_runtime,
                                "residual": np.nan,
                                "error_bound": error_bound,
                                "status": status,
                                "error": error,
                            }
                        )
                        if verbose:
                            message = f"{status} ({runtime:.3f} s)"
                            if error:
                                message += f" — {error}"
                            if validate:
                                message += f"; validation {validation_status}"
                            if validation_error:
                                message += f" — {validation_error}"
                            print(message, flush=True)
        finally:
            np.random.set_state(numpy_state)
            random.setstate(random_state)
        return BenchmarkResult(rows=rows, experiment=experiment)
