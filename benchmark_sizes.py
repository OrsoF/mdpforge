"""Measure catalogue models at small and medium sizes and append trials to CSV.

Run from the repository: conda run -n benchmark python benchmark_sizes.py
Edit STATE_DIM_REQUESTS to change individual recipes after examining the CSV.
Large sizes will be inferred later from these measurements, not benchmarked here.
Timings include solver construction/backend conversion and run(), but exclude
model copying, generation and independent Bellman-residual verification.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import random
import subprocess
import sys
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

ROOT = Path(__file__).resolve().parent
SIZES = ("small", "medium")
DEFAULT_STATE_DIMS = (100, 1000)
DEFAULT_SOLVERS = (
    "mdpforge_vi", "mdptoolbox_vi", "marmote_vi", "mdpsolver_vi"
)

# Constructor inputs, NOT guaranteed actual dimensions. None records an
# unavailable size rather than inventing a scaled version of a fixed model.
STATE_DIM_REQUESTS = {
    "access_control": (10000, 250000),
    "ambulance": (10000, 40000),
    "ambulance_relocation": (2500, 40000),
    "block": (100, 400),
    "blocks_world": (96, None),
    "car_rental": (625, 4096),
    "dam": (16, 81),
    "elevator": (100, 20000),
    "garnet": (100, 200),
    "hexagonal_grid_soccer": (343, 6859),
    "impatience": (100, 300),
    "inventory_leadtime": (100, 10000),
    "oil_discovery": (100, 300),
    "peg_solitaire": (None, 512),
    "sysadmin": (64, 512),
    "tandem": (100, 900),
    "tandem_choice": (36, 100),
    "wumpus_world": (512, 1024),
}

FIELDS = (
    "case_id", "model", "size", "requested_state_dim", "constructor_options",
    "model_seed", "source_sha256", "fingerprint", "state_dim", "action_dim",
    "nnz", "matrix_bytes", "density", "build_seconds", "solver", "repeat",
    "seed", "discount", "precision", "threads", "runtime", "wall_seconds",
    "timeout_seconds", "build_timeout_seconds", "max_states", "max_matrix_mib",
    "residual", "error_bound", "status", "error",
)


def build_model(request):
    import numpy as np

    module = import_module(f"mdpforge.models.{request['model']}")
    np.random.seed(request["model_seed"])
    random.seed(request["model_seed"])
    for key, generator in list(vars(module).items()):
        if isinstance(generator, np.random.Generator):
            setattr(module, key, np.random.Generator(
                type(generator.bit_generator)(request["model_seed"])
            ))
    options = json.loads(request["constructor_options"])
    if "custom_track" in options:
        options["custom_track"] = np.asarray(options["custom_track"])
    start = perf_counter()
    model = module.Model(**options)
    if model.state_dim > request["max_states"]:
        return {"status": "size_limit", "state_dim": int(model.state_dim),
                "action_dim": int(model.action_dim),
                "error": f"Actual states exceed --max-states={request['max_states']}"}
    # A distinct name prevents loading reference caches or stale model parameters.
    model.name = f"size_scan_{request['case_id']}_{model.name}"
    model.create_model(save=False)
    build_seconds = perf_counter() - start
    model.test_model()
    digest = hashlib.sha256()
    digest.update(f"{model.state_dim},{model.action_dim}".encode())
    arrays = [model.reward_matrix] + [
        getattr(matrix, field)
        for matrix in model.transition_matrix for field in ("data", "indices", "indptr")
    ]
    for array in arrays:
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    nnz = sum(matrix.nnz for matrix in model.transition_matrix)
    matrix_bytes = sum(array.nbytes for array in arrays)
    metadata = {
        "status": "success", "build_seconds": build_seconds,
        "state_dim": int(model.state_dim), "action_dim": int(model.action_dim),
        "nnz": int(nnz), "matrix_bytes": int(matrix_bytes),
        "density": nnz / (model.state_dim ** 2 * model.action_dim),
        "fingerprint": digest.hexdigest(),
    }
    if matrix_bytes > request["max_matrix_mib"] * 1024 ** 2:
        return {**metadata, "status": "memory_limit",
                "error": "Built matrices exceed --max-matrix-mib"}
    with Path(request["model_path"]).open("wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return metadata


def solve_model(request):
    from copy import deepcopy

    import numpy as np

    from mdpforge.core.operators import optimal_bellman_operator

    try:
        constructor = import_module(f"mdpforge.solvers.{request['solver']}").Solver
    except ModuleNotFoundError as exc:
        return {"status": "missing_dependency", "error": str(exc)}
    with Path(request["model_path"]).open("rb") as handle:
        reference = pickle.load(handle)
    trial = deepcopy(reference)
    np.random.seed(request["seed"])
    random.seed(request["seed"])
    start = perf_counter()
    try:
        solver = constructor(
            trial, request["discount"], final_precision=request["precision"]
        )
        solver.run()
    except Exception as exc:
        return {"status": "error", "runtime": perf_counter() - start,
                "error": f"{type(exc).__name__}: {exc}"}
    runtime = perf_counter() - start
    value = np.asarray(solver.value)
    if value.shape != (reference.state_dim,) or not np.all(np.isfinite(value)):
        return {"status": "invalid_value", "runtime": runtime,
                "error": "Value must be a finite vector of actual state_dim"}
    residual = float(np.max(np.abs(optimal_bellman_operator(
        reference, value, request["discount"]
    ) - value)))
    error_bound = residual / (1 - request["discount"])
    success = np.isfinite(error_bound) and error_bound <= request["precision"]
    return {
        "runtime": runtime, "residual": residual if np.isfinite(residual) else None,
        "error_bound": error_bound if np.isfinite(error_bound) else None,
        "status": "success" if success else "imprecise",
        "error": "" if success else "Bellman residual exceeds target precision",
    }


def worker(stage, request_path, result_path):
    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    try:
        result = build_model(request) if stage == "build" else solve_model(request)
    except Exception as exc:
        result = {"status": f"{stage}_error", "error": f"{type(exc).__name__}: {exc}"}
    Path(result_path).write_text(json.dumps(result, allow_nan=False), encoding="utf-8")


def run_worker(stage, request, folder, timeout):
    stem = "build" if stage == "build" else f"{request['solver']}_{request['repeat']}"
    input_path, result_path = folder / f"{stem}.input.json", folder / f"{stem}.json"
    input_path.write_text(json.dumps(request), encoding="utf-8")
    start = perf_counter()
    with (folder / f"{stem}.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--worker",
             stage, str(input_path), str(result_path)],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            env={**os.environ, "OPENBLAS_NUM_THREADS": str(request["threads"]),
                 "OMP_NUM_THREADS": str(request["threads"]),
                 "MKL_NUM_THREADS": str(request["threads"])},
        )
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"status": f"{stage}_timeout",
                    "wall_seconds": perf_counter() - start,
                    "error": f"Process exceeded {timeout:g} s"}
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    if process.returncode or not result_path.exists():
        return {"status": "process_error", "wall_seconds": perf_counter() - start,
                "error": f"Exit code {process.returncode}: " +
                (folder / f"{stem}.log").read_text(encoding="utf-8")[-2000:]}
    return {**json.loads(result_path.read_text(encoding="utf-8")),
            "wall_seconds": perf_counter() - start}


def main(args):
    from pkgutil import iter_modules

    from mdpforge import list_models, models, solvers

    catalogue = {entry["name"] for entry in list_models()}
    names = sorted(catalogue) if args.models is None else list(dict.fromkeys(args.models))
    args.solvers = list(dict.fromkeys(args.solvers))
    solver_names = {m.name for m in iter_modules(solvers.__path__)
                    if not m.ispkg and not m.name.startswith("_")}
    if set(names) - catalogue or set(args.solvers) - solver_names:
        raise ValueError("Unknown model or solver name; use catalogue module names")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    scratch = ROOT / "artifacts/tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    completed = set()
    if output.exists() and output.stat().st_size:
        with output.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(FIELDS):
                raise ValueError(
                    "Existing CSV has a different schema; choose another --output"
                )
            completed = {(r["case_id"], r["solver"], r["repeat"]) for r in reader}

    with output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        if handle.tell() == 0:
            writer.writeheader()

        def save(row):
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
            completed.add((
                row["case_id"], row.get("solver", ""), str(row.get("repeat", ""))
            ))
            label = f"{row['model']} / {row['size']} / {row.get('solver', 'build')}"
            seconds = row.get("runtime")
            timing = "" if seconds is None else f" ({seconds:.3f} s)"
            print(f"{label}: {row['status']}{timing}", flush=True)

        for name in names:
            source_hash = hashlib.sha256(
                (Path(models.__file__).parent / f"{name}.py").read_bytes()
            ).hexdigest()
            requests = STATE_DIM_REQUESTS.get(name, DEFAULT_STATE_DIMS)
            for size, requested in zip(SIZES, requests):
                options = {"state_dim": requested}
                if name == "sutton":
                    side = {"small": 3, "medium": 6}[size]
                    options["custom_track"] = [[1] * side for _ in range(side)]
                    options["custom_track"][-1][-1] = 0
                base = {
                    "model": name, "size": size,
                    "requested_state_dim": requested,
                    "constructor_options": json.dumps(options, sort_keys=True),
                    "model_seed": args.seed, "source_sha256": source_hash,
                    "discount": args.discount, "precision": args.precision,
                    "threads": args.threads,
                    "timeout_seconds": args.timeout,
                    "build_timeout_seconds": args.build_timeout,
                    "max_states": args.max_states,
                    "max_matrix_mib": args.max_matrix_mib,
                }
                base["case_id"] = hashlib.sha256(
                    json.dumps(base, sort_keys=True).encode()
                ).hexdigest()[:24]
                if (base["case_id"], "", "") in completed:
                    continue
                pending = [(solver, repeat) for solver in args.solvers
                           for repeat in range(1, args.repeats + 1)
                           if (base["case_id"], solver, str(repeat)) not in completed]
                if not pending:
                    continue
                if requested is None:
                    save({**base, "status": "unavailable_size",
                          "error": "No distinct usable size specified for this recipe"})
                    continue
                print(
                    f"Building {name} / {size} (request={requested})",
                    flush=True,
                )
                with TemporaryDirectory(prefix="size-scan-", dir=scratch) as directory:
                    folder = Path(directory)
                    request = {**base, "model_path": str(folder / "model.pkl"),
                               "max_states": args.max_states,
                               "max_matrix_mib": args.max_matrix_mib}
                    metadata = run_worker("build", request, folder, args.build_timeout)
                    if metadata["status"] != "success":
                        save({**base, **metadata})
                        continue
                    base.update({key: value for key, value in metadata.items()
                                 if key not in {"status", "wall_seconds"}})
                    for solver, repeat in pending:
                        trial = {**request, "solver": solver, "repeat": repeat,
                                 "seed": args.seed + repeat - 1}
                        result = run_worker("solve", trial, folder, args.timeout)
                        save({**base, "solver": solver, "repeat": repeat,
                              "seed": trial["seed"], **result})
    print(f"CSV: {output}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", help="Default: all current catalogue models"
    )
    parser.add_argument("--solvers", nargs="+", default=list(DEFAULT_SOLVERS))
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/results/model_sizes.csv"
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--precision", type=float, default=1e-3)
    parser.add_argument(
        "--timeout", type=float, default=60, help="Seconds per solver process"
    )
    parser.add_argument("--build-timeout", type=float, default=60)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-states", type=int, default=100000)
    parser.add_argument("--max-matrix-mib", type=float, default=256)
    parser.add_argument("--worker", nargs=3, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.worker:
        positive = (
            args.precision, args.timeout, args.build_timeout, args.max_matrix_mib
        )
        if (not math.isfinite(args.discount) or not 0 < args.discount < 1
            or any(not math.isfinite(x) or x <= 0 for x in positive)
            or min(args.repeats, args.threads, args.max_states) < 1
            or not 0 <= args.seed < 2**32 - args.repeats):
            parser.error("Invalid discount, seed, precision, counts or limits")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.worker:
        worker(*arguments.worker)
    else:
        try:
            main(arguments)
        except KeyboardInterrupt:
            print("Interrupted; completed rows remain in the CSV. Rerun to resume.")
            sys.exit(130)
