import csv
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from core.operators import (
    bellman_operator,
    compute_transition_reward_policy,
    optimal_bellman_operator,
)
from solvers.aggregated_vi import Solver as AggregatedVI
from solvers.personal_vi import Solver as VI


@pytest.mark.parametrize("solver_class", [VI, AggregatedVI], ids=["vi", "pdvi"])
def test_model_to_solver_workflow(model_spec, isolated_model_cache, solver_class):
    model_class, state_dim, action_dim, _ = model_spec
    model = model_class(state_dim, action_dim)
    model.create_model(save=False)
    discount = 0.9
    solver = solver_class(model, discount, final_precision=1e-4)
    solver.run()
    assert solver.value.shape == (model.state_dim,)
    assert np.all(np.isfinite(solver.value))
    policy = bellman_operator(model, solver.value, discount).argmax(axis=1)
    if solver_class is VI:
        assert solver.bellman_residual() < 1e-4
    else:
        assert solver.policy.shape == (model.state_dim,)
        assert np.issubdtype(solver.policy.dtype, np.integer)
        assert np.all((0 <= solver.policy) & (solver.policy < model.action_dim))
        np.testing.assert_array_equal(solver.policy, policy)
    transition, reward = compute_transition_reward_policy(model, policy)
    policy_value = np.linalg.solve(
        np.eye(model.state_dim) - discount * transition.toarray(), reward
    )
    # A residual of 1e-4 implies a value error of at most 1e-3 at discount 0.9.
    assert_allclose(solver.value, policy_value, atol=1e-3, rtol=0)
    assert_allclose(
        optimal_bellman_operator(model, solver.value, discount),
        solver.value,
        atol=1e-4,
        rtol=0,
    )
    assert np.isfinite(solver.runtime) and solver.runtime >= 0


@pytest.mark.parametrize("script", ["main.py", "main_total.py"])
def test_documented_benchmark_cli_writes_runtime_tables(tmp_path, script):
    root = Path(__file__).resolve().parents[1]
    # Keep CLI caches and results isolated from the user's research artifacts.
    for directory in ("core", "models", "solvers", "utils"):
        shutil.copytree(
            root / directory,
            tmp_path / directory,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    shutil.copy2(root / script, tmp_path / script)
    output = tmp_path / "artifacts" / "tmp" / "demo"
    command = [
        sys.executable,
        script,
        "--model",
        "rooms",
        "--state",
        "100",
        "--solvers",
        "vi",
        "pdvi",
        "--repeat",
        "2",
        "--seed",
        "0",
        "--output-dir",
        str(output),
    ]
    if script == "main.py":
        command += ["--discount", "0.9"]
    result = subprocess.run(
        command, cwd=tmp_path, capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "|S|=100, |A|=4, runs=2" in result.stdout
    with (output / "rooms_runs.csv").open(newline="", encoding="utf-8") as handle:
        runs = list(csv.DictReader(handle))
    assert {(row["solver"], row["run"]) for row in runs} == {
        (solver, run) for solver in ("vi", "pdvi") for run in ("1", "2")
    }
    assert len(runs) == 4
    assert all(float(row["runtime"]) >= 0 for row in runs)
    with (output / "rooms.csv").open(newline="", encoding="utf-8") as handle:
        summary = list(csv.DictReader(handle))
    assert {row["solver"] for row in summary} == {"vi", "pdvi"}
    assert all(row["runs"] == "2" for row in summary)
    assert "\\begin{tabular}" in (output / "rooms.tex").read_text(encoding="utf-8")
