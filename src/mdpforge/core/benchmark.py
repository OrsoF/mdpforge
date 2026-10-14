"""Benchmark finite discounted MDP solvers at a fixed value precision."""

import inspect
import multiprocessing
import pickle
import random
import warnings
from copy import deepcopy
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from pkgutil import iter_modules
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np

from mdpforge import models, solvers
from mdpforge.core.mdp import MDP
from mdpforge.core.model import MDPProtocol
from mdpforge.core.operators import optimal_bellman_operator
from mdpforge.core.result import BenchmarkResult
from mdpforge.core.validation import validate_model
from mdpforge.solvers import list_solvers
from mdpforge.utils.experiment import capture_experiment


def _catalogue(package):
    """List public catalogue modules without importing their dependencies."""
    return {
        module.name
        for module in iter_modules(package.__path__)
        if not module.ispkg and not module.name.startswith("_")
    }


def _load_solver(name, *, backend=None):
    """Load a catalogue solver, checking an explicitly selected backend first."""
    dependency = {
        "gurobi": "gurobipy",
        "marmote": "marmote",
        "mdpsolver": "mdpsolver",
        "mdptoolbox": "mdptoolbox",
    }.get(backend)
    if dependency and find_spec(dependency) is None:
        raise ModuleNotFoundError(
            f"Missing solver dependency: {dependency}", name=dependency
        )
    return import_module(f"{solvers.__name__}.{name}").Solver


def _available_solvers(entries):
    """Yield selected solvers, warning only for missing dependencies."""
    for entry in entries:
        name = entry["name"]
        try:
            constructor = _load_solver(name, backend=entry.get("backend"))
        except ModuleNotFoundError as exc:
            warnings.warn(
                f"Skipping solver {name!r}: missing dependency {exc.name!r}",
                stacklevel=3,
            )
            continue
        yield name, constructor


def _check_solver_options(name, constructor, options):
    """Check options against the class/function calling convention used by run()."""
    try:
        signature = inspect.signature(constructor)
        if inspect.isclass(constructor):
            signature.bind(None, None, final_precision=None, **options)
        else:
            signature.bind(None, None, None, None, **options)
    except TypeError as exc:
        raise TypeError(f"Invalid options for solver {name!r}: {exc}") from exc


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


def _solver_trial(model, constructor, options, discount, precision, seed):
    """Return values, solver runtime, status and error for one independent trial."""
    runtime = np.nan
    start = None
    try:
        trial_model = deepcopy(model)
        np.random.seed(seed)
        random.seed(seed)
        start = perf_counter()
        if isinstance(constructor, type):
            solver = constructor(
                trial_model, discount, final_precision=precision, **options
            )
            solver.run()
            value = solver.value
        else:
            value = constructor(
                trial_model.transition_matrix,
                trial_model.reward_matrix,
                discount,
                precision,
                **options,
            )
        runtime = perf_counter() - start
        return np.asarray(value), runtime, "success", ""
    except Exception as exc:
        if start is not None and np.isnan(runtime):
            runtime = perf_counter() - start
        return None, runtime, "error", f"{type(exc).__name__}: {exc}"


def _solver_worker(input_path, result_path):
    # Files avoid blocking on a full IPC pipe during dispatch or worker exit.
    with input_path.open("rb") as handle:
        args = pickle.load(handle)
    with result_path.open("wb") as handle:
        pickle.dump(_solver_trial(*args), handle)


def _isolated_solver_trial(
    model, constructor, options, discount, precision, seed, timeout
):
    context = multiprocessing.get_context("spawn")
    with TemporaryDirectory(prefix="mdpforge-trial-") as directory:
        # Serialize before spawning so unpickleable inputs cannot leave a worker
        # behind if process startup fails halfway through transmitting its inputs.
        input_path = Path(directory) / "input.pkl"
        with input_path.open("wb") as handle:
            pickle.dump(
                (model, constructor, options, discount, precision, seed), handle
            )
        result_path = Path(directory) / "result.pkl"
        process = context.Process(
            target=_solver_worker,
            args=(input_path, result_path),
        )
        try:
            start = perf_counter()
            process.start()
            process.join(max(0, timeout - (perf_counter() - start)))
            if process.is_alive():
                return (
                    None,
                    np.nan,
                    "timeout",
                    f"Trial exceeded timeout of {timeout:g} s",
                )
            if process.exitcode != 0:
                return (
                    None,
                    np.nan,
                    "error",
                    f"Solver process exited with code {process.exitcode}",
                )
            with result_path.open("rb") as handle:
                return pickle.load(handle)
        finally:
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                    process.join(1)
                if process.is_alive():
                    process.kill()
                    process.join(1)
                if not process.is_alive():
                    process.close()


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
        self._default_solvers = default_solvers
        self._mdps: list[MDPProtocol] = []
        self._solvers = {}  # label -> (solver class/function, options)
        if default_solvers:
            from mdpforge.solvers.mdpforge_vi import Solver as VI

            self.add_solver("VI", solve_function=VI)

    def reset(self, *, models=True, solvers=True):
        """Reset registrations; previously returned BenchmarkResult objects are kept.

        models: True removes all registered MDPs; False keeps them.
        solvers: True restores VI alone if constructed with default_solvers=True,
            otherwise removes all solvers. False keeps solvers and their options.

        Return self for chaining. Cached models and exported files are kept.
        """
        if models:
            self._mdps.clear()
        if solvers:
            self._solvers.clear()
            if self._default_solvers:
                from mdpforge.solvers.mdpforge_vi import Solver as VI

                self.add_solver("VI", solve_function=VI)
        return self

    def add_mdp(
        self,
        model: MDPProtocol | str | None = None,
        *,
        transitions=None,
        reward=None,
        size: str | None = None,
        category: str | None = None,
    ):
        """Add named, matrix-defined or built MDPs, or filtered groups; return self.

        model: catalogue ID from list_models(), built MDP object, or custom name
            with transitions and reward. None selects by size/category filters.
        transitions: A dense/sparse (S, S) matrices, one per action;
            None for catalogue models or built objects. S = states, A = actions.
        reward: dense/sparse (S, A) expected rewards, indexed [state, action];
            None for catalogue models or built objects.
        size: "small", "medium" or "large" preset;
            None uses recipe defaults and does not filter available sizes.
        category: "abstract", "navigation", "physical_control",
            "service_operations", "asset_management" or "games_and_puzzles".
            None accepts all task families.

        size/category accept single strings and combine with AND. Filters cannot
        accompany an explicit model or matrices. Custom names must be unique.
        Groups skip unavailable presets, duplicates and missing dependencies.
        Provide a model or filter; use add_all_models() for the entire catalogue.
        """
        if size is not None or category is not None:
            if model is not None or transitions is not None or reward is not None:
                raise ValueError(
                    "size/category filters cannot be combined with a model or matrices."
                )
            for entry in models.list_models(size=size, category=category):
                preset = entry["sizes"][size] if size is not None else None
                self._add_catalogue_model(entry["name"], preset=preset, grouped=True)
            return self
        if model is None:
            raise ValueError(
                "Provide a model or at least one size/category filter; "
                "use add_all_models() to add the entire catalogue."
            )
        if isinstance(model, str):
            if not model.strip():
                raise ValueError("Each MDP must have a nonempty string 'name'.")
            catalogue = {entry["name"] for entry in models.list_models()}
            if model in catalogue:
                if transitions is not None or reward is not None:
                    raise ValueError(
                        f"Catalogue model {model!r} uses its generator; "
                        "choose a custom name when supplying matrices."
                    )
                return self._add_catalogue_model(model)
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
        return self

    def _add_catalogue_model(self, name, *, preset=None, grouped=False):
        """Build a recipe; groups skip duplicates and missing dependencies."""
        try:
            options = dict(preset["parameters"]) if preset is not None else {}
            if "custom_track" in options:
                options["custom_track"] = np.asarray(options["custom_track"]).copy()
            model = import_module(f"{models.__name__}.{name}").Model(**options)
            if any(existing.name == model.name for existing in self._mdps):
                if grouped:
                    return self
                raise ValueError(f"An MDP named {model.name!r} is already registered.")
            model.create_model()
            if preset is not None and model.state_dim != preset["state_dim"]:
                raise ValueError(
                    f"{name}: expected {preset['state_dim']} states, "
                    f"got {model.state_dim}; check the preset and cached matrices"
                )
            return self.add_mdp(model)
        except ModuleNotFoundError as exc:
            if not grouped:
                raise
            warnings.warn(
                f"Skipping model {name!r}: missing dependency {exc.name!r}",
                stacklevel=3,
            )
            return self

    def add_all_models(self):
        """Add all catalogue models with default dimensions, without duplicates.

        Missing dependencies are skipped with a warning; other errors propagate.
        """
        for entry in models.list_models():
            self._add_catalogue_model(entry["name"], grouped=True)
        return self

    def add_solver(
        self,
        solver_name: str | None = None,
        *,
        solve_function=None,
        family=None,
        backend=None,
        update_scheme=None,
        **options,
    ):
        """Add solvers by name, filters or custom callable; return self.

        solver_name: catalogue ID from list_solvers(), or a custom label.
            None selects a group using the filters below.
        solve_function: custom function/class, or None for catalogue solvers.
            Requires solver_name; cannot accompany metadata filters.
        family: "vi" (value iteration), "pi" (policy iteration),
            "mpi" (modified policy iteration), "lp" (linear programming).
        backend: implementation library: "mdpforge", "mdpsolver", "marmote",
            "mdptoolbox" or "gurobi".
        update_scheme:
            "synchronous": full-vector Bellman sweeps.
            "gauss_seidel": in-place state updates.
            "sor": successive over-relaxation.
            "linear_solve": matrix-based policy evaluation.
            "mixed": Bellman sweeps with aggregate or stochastic corrections.
            "not_applicable": LP, without Bellman sweeps.
        **options: selected solver's constructor/function keywords, e.g.
            parallel=True or initial_value=array. See its signature for options;
            problem data, discount and precision are supplied by run().

        Filters accept a string, nonempty sequence of strings, or None (ignored).
        Fields combine with AND; values within each sequence combine with OR.
        """
        filters = {
            "family": family,
            "backend": backend,
            "update_scheme": update_scheme,
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
        if not grouped and solver_name in self._solvers:
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
        if grouped:
            return self._add_solver_group(list_solvers(**filters), options)
        if solve_function is None:
            catalogue = _catalogue(solvers)
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
            solve_function = _load_solver(solver_name)
        if not callable(solve_function):
            raise TypeError("solve_function must be a callable function or class.")
        self._solvers[solver_name] = (solve_function, options)
        return self

    def _add_solver_group(self, entries, options):
        """Check the entire selection before committing new registrations."""
        pending = {}
        for name, constructor in _available_solvers(entries):
            _check_solver_options(name, constructor, options)
            if name in self._solvers:
                existing, previous_options = self._solvers[name]
                if existing is constructor and _same_options(previous_options, options):
                    continue
                raise ValueError(
                    f"Solver {name!r} is already registered with a different "
                    "implementation or configuration"
                )
            if any(
                existing is constructor and _same_options(previous_options, options)
                for existing, previous_options in (
                    *self._solvers.values(),
                    *pending.values(),
                )
            ):
                continue
            pending[name] = (constructor, options.copy())
        if pending:
            self._solvers.update(pending)
        return self

    def add_all_solvers(self):
        """Add all catalogue solvers with defaults, preserving existing entries.

        Default VI is not duplicated under its catalogue name. Missing
        dependencies are skipped with a warning; other errors propagate.
        """
        entries = (
            {"name": name}
            for name in sorted(_catalogue(solvers))
            if name not in self._solvers
        )
        for name, constructor in _available_solvers(entries):
            if any(
                existing is constructor and not options
                for existing, options in self._solvers.values()
            ):
                continue
            self.add_solver(name, solve_function=constructor)
        return self

    def run(
        self,
        discount,
        precision=1e-3,
        *,
        repeats=3,
        seed=0,
        verbose=False,
        timeout=None,
    ) -> BenchmarkResult:
        """Run each MDP/solver pair and return an independent BenchmarkResult.

        ``precision`` is an upper bound on ||V - V*||_infinity, certified by
        ||T V - V||_infinity / (1 - discount). All failures are recorded rather
        than interrupting the remaining comparisons.

        ``verbose=True`` prints trial progress, runtime and failure details
        outside solver timing.

        Each solver receives an independent copy of its MDP. Seeds are paired
        across solvers and MDPs for each repeat. Global NumPy and Python RNG
        states are restored; solvers using private RNGs must seed those
        generators themselves.

        ``timeout`` is an optional positive wall-clock limit in seconds per
        trial. It uses a fresh spawn process and covers child startup, model
        copying, solver construction/execution and result serialization, after
        dispatch from the parent. Parent input serialization and final precision
        verification are outside this limit. Process cleanup can take up to two
        extra seconds. Timed-out or crashed trials have no solver runtime (NaN).
        The parent still verifies values against the original MDP.

        With a timeout, models, solvers and options must be pickleable; custom
        functions/classes must be importable from a module. In scripts, protect
        the benchmark call with ``if __name__ == "__main__":``. Notebook-local
        callables are unsupported in this mode. Only the dedicated worker is
        terminated, not any descendant processes started by a custom solver.

        The result's ``experiment`` captures fingerprints, versions and
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

        for model in self._mdps:
            validate_model(model)

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
            )
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
                        try:
                            args = (
                                model,
                                constructor,
                                kwargs,
                                discount,
                                precision,
                                seed + repeat,
                            )
                            if timeout is None:
                                value, runtime, status, error = _solver_trial(*args)
                            else:
                                value, runtime, status, error = _isolated_solver_trial(
                                    *args, timeout
                                )
                            if status == "success":
                                if value.shape != (model.state_dim,) or not np.all(
                                    np.isfinite(value)
                                ):
                                    raise ValueError(
                                        "Solver values must be a finite vector "
                                        "of state_dim"
                                    )

                                residual = float(
                                    np.max(
                                        np.abs(
                                            optimal_bellman_operator(
                                                model, value, discount
                                            )
                                            - value
                                        )
                                    )
                                )
                                error_bound = residual / (1 - discount)
                                status = (
                                    "success"
                                    if error_bound <= precision
                                    else "imprecise"
                                )
                                if status == "imprecise":
                                    error = (
                                        f"Value error bound {error_bound:g} "
                                        f"exceeds precision {precision:g}"
                                    )
                        except Exception as exc:
                            status = "error"
                            error = f"{type(exc).__name__}: {exc}"
                        rows.append(
                            {
                                "experiment_id": experiment["experiment_id"],
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
        return BenchmarkResult(rows=rows, experiment=experiment)
