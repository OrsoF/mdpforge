"""Capture experiment provenance and execute independent solver trials."""

import inspect
import json
import math
import multiprocessing
import os
import pickle
import platform
import random
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from uuid import uuid4

import numpy as np
import scipy
from scipy.sparse import issparse
from threadpoolctl import threadpool_info

from mdpforge.core.solve_result import SolveResult
from mdpforge.solvers import is_catalogue_solver


def _qualified_name(value):
    target = (
        value if inspect.isclass(value) or inspect.isroutine(value) else type(value)
    )
    return f"{target.__module__}.{target.__qualname__}"


def _json_value(value):
    if isinstance(value, np.generic):
        scalar = value.item()
        if isinstance(scalar, np.generic):
            return {"dtype": str(value.dtype), "value": str(value)}
        return _json_value(scalar)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"nonfinite_float": str(value)}
    if isinstance(value, np.ndarray):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": _json_value(value.tolist()),
        }
    if issparse(value):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "nnz": int(value.nnz),
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        # Non-string keys occur in model state encodings; retain their types.
        if all(isinstance(key, str) for key in value):
            return {key: _json_value(item) for key, item in value.items()}
        return {
            "entries": [
                [_json_value(key), _json_value(item)] for key, item in value.items()
            ]
        }
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(
            (_json_value(item) for item in value),
            key=lambda item: json.dumps(item, sort_keys=True),
        )
    if callable(value):
        return {"callable": _qualified_name(value)}
    return {"unserialized_type": _qualified_name(value)}


def model_config(model):
    """Serialize stored constructor parameters; models may override get_config()."""
    try:
        parameters = inspect.signature(type(model)).parameters
    except (TypeError, ValueError):
        return {}
    return {
        name: _json_value(getattr(model, f"requested_{name}", getattr(model, name)))
        for name, parameter in parameters.items()
        if parameter.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        and hasattr(model, name)
    }


def _solver_metadata(constructor, options):
    module = inspect.getmodule(constructor)
    metadata = (
        getattr(module, "METADATA", {})
        if module is not None and getattr(module, "Solver", None) is constructor
        else {}
    )
    try:
        parameters = inspect.signature(constructor).parameters
        positional = [
            name
            for name, parameter in parameters.items()
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        # Match Benchmark's calling convention, regardless of argument names.
        input_count = 2 if inspect.isclass(constructor) else 4
        controlled = set(positional[:input_count])
        if inspect.isclass(constructor):
            controlled.add("final_precision")
        defaults = {
            name: parameter.default
            for name, parameter in parameters.items()
            if name not in controlled
            and parameter.default is not inspect.Parameter.empty
        }
    except (TypeError, ValueError):
        defaults = {}
    return {
        "callable": _qualified_name(constructor),
        "options": _json_value(options),
        "parameters": _json_value(defaults | options),
        "metadata": _json_value(metadata),
    }


def _package_version(distribution):
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _backend_versions(solvers):
    names = {record["metadata"].get("backend") for record in solvers.values()} - {
        None,
        "mdpforge",
    }
    aliases = {"mdptoolbox": "pymdptoolbox"}
    backends = {}
    for name in sorted(names):
        distribution = aliases.get(name, name)
        backends[name] = {
            "distribution": distribution,
            "version": _package_version(distribution),
        }
    return backends


def _git_metadata():
    def git(*arguments):
        return subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), *arguments],
            capture_output=True,
            check=True,
            timeout=5,
        ).stdout

    try:
        return {
            "commit": git("rev-parse", "HEAD").decode().strip(),
            "dirty": bool(git("status", "--porcelain")),
        }
    except (OSError, subprocess.SubprocessError):
        return None


def _machine_metadata():
    thread_variables = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
        "thread_environment": {name: os.environ.get(name) for name in thread_variables},
        "thread_pools": threadpool_info(),
        "capture_scope": "benchmark process before solver execution",
        "numpy_build": _json_value(getattr(np.__config__, "CONFIG", None)),
    }


def capture_experiment(
    models,
    solvers,
    discount,
    precision,
    repeats,
    seed,
    *,
    timeout=None,
    validate=True,
    validation_methods=None,
):
    """Freeze metadata once per run, before any solver is constructed."""
    model_metadata = {}
    for model in models:
        get_config = getattr(model, "get_config", None)
        model_metadata[model.name] = {
            "class": _qualified_name(model),
            "display_name": getattr(model, "display_name", model.name),
            "state_dim": int(model.state_dim),
            "requested_state_dim": _json_value(
                getattr(model, "requested_state_dim", None)
            ),
            "action_dim": int(model.action_dim),
            "nnz": sum(matrix.nnz for matrix in model.transition_matrix),
            "config": _json_value(
                get_config() if callable(get_config) else model_config(model)
            ),
        }
    solver_metadata = {
        name: _solver_metadata(constructor, options)
        for name, (constructor, options) in solvers.items()
    }
    for name, method in (validation_methods or {}).items():
        solver_metadata[name]["validation_method"] = method
    return {
        "schema_version": 2,
        "experiment_id": str(uuid4()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "discount": float(discount),
            "precision": float(precision),
            "repeats": repeats,
            "seed": seed,
            "trial_seeds": list(range(seed, seed + repeats)),
            "seed_scope": "solver trials; model generation is not reseeded",
            "timeout": None if timeout is None else float(timeout),
            "execution_mode": "in_process" if timeout is None else "spawn",
            "validate": validate,
        },
        "models": model_metadata,
        "solvers": solver_metadata,
        "environment": {
            "python": platform.python_version(),
            "mdpforge": _package_version("mdpforge"),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "threadpoolctl": _package_version("threadpoolctl"),
            "backends": _backend_versions(solver_metadata),
        },
        "machine": _machine_metadata(),
        "code": {"git": _git_metadata()},
    }


def _solver_signature(constructor, precision):
    """Resolve the constructor signature and benchmark-controlled precision option."""
    signature = inspect.signature(constructor)
    precision_options = {}
    if isinstance(constructor, type) and (
        "final_precision" in signature.parameters
        or any(p.kind == p.VAR_KEYWORD for p in signature.parameters.values())
    ):
        precision_options["final_precision"] = precision
    return signature, precision_options


def _check_solver_options(name, constructor, options):
    """Check options against the benchmark's class or function calling convention."""
    try:
        signature, precision_options = _solver_signature(constructor, None)
        if isinstance(constructor, type):
            signature.bind(None, None, **precision_options, **options)
        else:
            signature.bind(None, None, None, None, **options)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Invalid options for solver {name!r}: {exc}") from exc


def _solver_trial(model, constructor, options, discount, precision, seed, solver_name):
    """Return the full result, solver runtime, status and error for one trial."""
    runtime = np.nan
    start = None
    try:
        trial_model = deepcopy(model)
        np.random.seed(seed)
        random.seed(seed)
        precision_options = {}
        if isinstance(constructor, type):
            _, precision_options = _solver_signature(constructor, precision)
        start = perf_counter()
        if isinstance(constructor, type):
            solver = constructor(trial_model, discount, **precision_options, **options)
            if is_catalogue_solver(constructor):
                solver.name = solver_name
            result = solver.run()
        else:
            result = constructor(
                trial_model.transition_matrix,
                trial_model.reward_matrix,
                discount,
                precision,
                **options,
            )
        runtime = perf_counter() - start
        if not isinstance(constructor, type) and not isinstance(result, SolveResult):
            result = SolveResult(np.asarray(result), None, validation_method="residual")
        return result, runtime, "success", ""
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
    model, constructor, options, discount, precision, seed, solver_name, timeout
):
    context = multiprocessing.get_context("spawn")
    with TemporaryDirectory(prefix="mdpforge-trial-") as directory:
        # Serialize before spawning so unpickleable inputs cannot leave a worker
        # behind if process startup fails halfway through transmitting its inputs.
        input_path = Path(directory) / "input.pkl"
        with input_path.open("wb") as handle:
            pickle.dump(
                (model, constructor, options, discount, precision, seed, solver_name),
                handle,
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
