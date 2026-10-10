"""Validate current catalogue size presets, optionally measuring solver runtimes.

Quick audit: conda run -n benchmark python check_model_sizes.py --dimensions-only
Runtime validation: conda run -n benchmark python check_model_sizes.py --sizes large
Uses metadata in model files, not the historical inferred_environment_sizes.json.
Each constructor/build/solver runs in a timed subprocess; results append to CSV.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmark_sizes import DEFAULT_SOLVERS, ROOT, run_worker

BUDGETS = {"small": 1.0, "medium": 5.0, "large": 30.0}
FIELDS = [
    "case_id",
    "model",
    "size",
    "preset_source",
    "expected_state_dim",
    "state_dim",
    "action_dim",
    "constructor_options",
    "source_sha256",
    "mode",
    "model_seed",
    "discount",
    "precision",
    "threads",
    "timeout_seconds",
    "build_timeout_seconds",
    "max_states",
    "max_matrix_mib",
    "solver",
    "repeat",
    "seed",
    "status",
    "build_seconds",
    "nnz",
    "matrix_bytes",
    "fingerprint",
    "runtime",
    "wall_seconds",
    "residual",
    "error_bound",
    "budget_seconds",
    "budget_met",
    "error",
]


def evaluate(case, args, completed):
    """Yield rows for one preset; never read/write the reference model caches."""
    if case["constructor_options"] is None:
        if (case["case_id"], "", "") not in completed:
            yield {**case, "status": "unavailable_size", "error": case["error"]}
        return
    if args.dimensions_only or args.build_only:
        pending = [("", "")]
    else:
        pending = [
            (solver, str(repeat))
            for solver in args.solvers
            for repeat in range(1, args.repeats + 1)
        ]
    pending = [pair for pair in pending if (case["case_id"], *pair) not in completed]
    if not pending or (case["case_id"], "", "") in completed:
        return
    scratch = ROOT / "artifacts/tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="preset-check-", dir=scratch) as directory:
        folder = Path(directory)
        request = {
            **case,
            "dimensions_only": args.dimensions_only,
            "model_path": str(folder / "model.pkl"),
        }
        metadata = run_worker("build", request, folder, args.build_timeout)
        if metadata["status"] != "success" or args.dimensions_only or args.build_only:
            yield {**case, **metadata}
            return
        base = {
            **case,
            **{k: v for k, v in metadata.items() if k not in {"status", "wall_seconds"}},
        }
        for solver, repeat in pending:
            trial = {
                **request,
                "solver": solver,
                "repeat": int(repeat),
                "seed": args.seed + int(repeat) - 1,
            }
            result = run_worker("solve", trial, folder, args.timeout)
            success = result["status"] == "success"
            yield {
                **base,
                "solver": solver,
                "repeat": repeat,
                "seed": trial["seed"],
                **result,
                "budget_met": success and result["runtime"] <= case["budget_seconds"],
            }


def main(args):
    from pkgutil import iter_modules

    from mdpforge import list_models, models, solvers

    catalogue = {entry["name"]: entry for entry in list_models()}
    names = sorted(catalogue) if args.models is None else list(dict.fromkeys(args.models))
    if set(names) - catalogue.keys():
        raise ValueError(f"Unknown models: {sorted(set(names) - catalogue.keys())}")
    available_solvers = {
        module.name
        for module in iter_modules(solvers.__path__)
        if not module.ispkg and not module.name.startswith("_")
    }
    if set(args.solvers) - available_solvers:
        raise ValueError(f"Unknown solvers: {sorted(set(args.solvers) - available_solvers)}")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    statuses = Counter()
    problems = set()

    def account(row):
        statuses[row["status"]] += 1
        if row["status"] != "success" or str(row.get("budget_met", "")) == "False":
            problems.add(f"{row['model']}/{row['size']}")

    cases = []
    for name in names:
        source_hash = hashlib.sha256(
            (Path(models.__file__).parent / f"{name}.py").read_bytes()
        ).hexdigest()
        for size in args.sizes:
            preset = catalogue[name]["sizes"].get(size, {})
            parameters = preset.get("parameters")
            if parameters is not None and (
                not isinstance(preset.get("state_dim"), int) or preset["state_dim"] <= 0
            ):
                raise ValueError(f"{name}/{size}: missing or invalid expected state_dim")
            case = {
                "model": name,
                "size": size,
                "preset_source": preset.get("source", "unavailable"),
                "expected_state_dim": preset.get("state_dim"),
                "constructor_options": (
                    json.dumps(parameters, sort_keys=True)
                    if parameters is not None
                    else None
                ),
                "source_sha256": source_hash,
                "mode": (
                    "dimensions"
                    if args.dimensions_only
                    else "build" if args.build_only else "solve"
                ),
                "model_seed": args.seed,
                "discount": args.discount,
                "precision": args.precision,
                "threads": args.threads,
                "timeout_seconds": args.timeout,
                "build_timeout_seconds": args.build_timeout,
                "max_states": args.max_states,
                "max_matrix_mib": args.max_matrix_mib,
                "budget_seconds": BUDGETS[size],
                "error": (
                    preset.get("reason", "No preset declared")
                    if parameters is None
                    else ""
                ),
            }
            case["case_id"] = hashlib.sha256(
                json.dumps(case, sort_keys=True).encode()
            ).hexdigest()[:24]
            cases.append(case)
    selected = {case["case_id"] for case in cases}
    if output.exists() and output.stat().st_size:
        with output.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != FIELDS:
                raise ValueError("Different CSV schema; choose another --output")
            for row in reader:
                completed.add((row["case_id"], row["solver"], row["repeat"]))
                if row["case_id"] in selected and (
                    not row["solver"]
                    or (row["solver"] in args.solvers and int(row["repeat"]) <= args.repeats)
                ):
                    account(row)
    with output.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        if handle.tell() == 0:
            writer.writeheader()

        def save(row):
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
            account(row)
            timing = (
                f" ({row['runtime']:.3f} s)" if row.get("runtime") is not None else ""
            )
            label = row.get("solver") or row["mode"]
            budget = (
                " ABOVE BUDGET"
                if row.get("budget_met") is False and row["status"] == "success"
                else ""
            )
            print(
                f"{row['model']}/{row['size']}/{label}: {row['status']}{timing}{budget}",
                flush=True,
            )

        # Constructor audits can run independently. Full solver measurements are
        # sequential so that concurrent jobs do not distort runtime comparisons.
        if args.dimensions_only:
            executor = ThreadPoolExecutor(max_workers=args.jobs)
            try:
                results = executor.map(
                    lambda case: list(evaluate(case, args, completed)), cases
                )
                for rows in results:
                    for row in rows:
                        save(row)
            finally:
                executor.shutdown(wait=True, cancel_futures=True)
        else:
            for case in cases:
                for row in evaluate(case, args, completed):
                    save(row)
    print(
        f"Summary: {dict(statuses)}\n"
        f"Problems: {', '.join(sorted(problems)) or 'none'}\nCSV: {output}"
    )
    return 1 if problems else 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--sizes", nargs="+", choices=tuple(BUDGETS))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dimensions-only", action="store_true")
    modes.add_argument("--build-only", action="store_true")
    parser.add_argument("--solvers", nargs="+", default=list(DEFAULT_SOLVERS))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--precision", type=float, default=1e-3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--build-timeout", type=float, default=60)
    parser.add_argument("--max-states", type=int, default=100000)
    parser.add_argument("--max-matrix-mib", type=float, default=256)
    parser.add_argument("--jobs", type=int, default=4, help="Parallel constructor checks only")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.sizes = list(
        dict.fromkeys(
            args.sizes or (list(BUDGETS) if args.dimensions_only else ["large"])
        )
    )
    args.solvers = list(dict.fromkeys(args.solvers))
    suffix = (
        "dimensions" if args.dimensions_only else "build" if args.build_only else "runtimes"
    )
    args.output = args.output or ROOT / f"artifacts/results/model_size_{suffix}.csv"
    if (
        not math.isfinite(args.discount)
        or not 0 < args.discount < 1
        or any(
            not math.isfinite(x) or x <= 0
            for x in (
                args.precision, args.timeout, args.build_timeout, args.max_matrix_mib
            )
        )
        or min(args.repeats, args.threads, args.jobs, args.max_states) < 1
        or not 0 <= args.seed < 2**32 - args.repeats
    ):
        parser.error("Invalid discount, seed, precision, counts or limits")
    return args


if __name__ == "__main__":
    try:
        raise SystemExit(main(parse_args()))
    except KeyboardInterrupt:
        print("Interrupted; completed rows remain in CSV. Rerun to resume.")
        raise SystemExit(130)
