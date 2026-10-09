import csv
import multiprocessing
import os
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from scipy.sparse import eye

from mdpforge import Benchmark
from mdpforge.core.mdp import MDP


def function_trial(transitions, rewards, discount, precision, *, outcome, marker=None):
    if outcome == "block":
        Path(marker).write_text(str(os.getpid()))
        while True:
            time.sleep(1)
    if outcome == "crash":
        os._exit(23)
    if outcome == "exception":
        raise RuntimeError("trial failed")
    if outcome == "invalid":
        return np.zeros(rewards.shape[0] - 1)
    if outcome == "mutate":
        rewards[:] = 0
        for matrix in transitions:
            matrix.data[:] = 0
        return np.zeros(rewards.shape[0])
    if outcome == "imprecise":
        return np.zeros(rewards.shape[0])
    # Known optimal values on the deterministic chain; a seeded uniform offset
    # exposes both NumPy and Python RNG behavior through the certified residual.
    offset = precision * 0.01 * (np.random.random() + random.random())
    return np.array([2 * discount, 2, 0]) + offset


class ClassTrial:
    def __init__(self, model, discount, final_precision, *, outcome, marker=None):
        self.model = model
        self.discount = discount
        self.precision = final_precision
        self.outcome = outcome
        self.marker = marker
        if outcome == "block_constructor":
            function_trial(None, None, None, None, outcome="block", marker=marker)

    def run(self):
        self.value = function_trial(
            self.model.transition_matrix,
            self.model.reward_matrix,
            self.discount,
            self.precision,
            outcome=self.outcome,
            marker=self.marker,
        )


def constant_reward_trial(transitions, rewards, discount, precision):
    return rewards[:, 0] / (1 - discount)


@pytest.fixture
def matrix_chain(chain):
    # Module-defined model and solvers work with spawn on Windows and POSIX.
    return MDP.from_matrices(
        "timeout_chain", chain.transition_matrix, chain.reward_matrix
    )


@pytest.mark.parametrize(
    "constructor, outcome",
    [
        (function_trial, "block"),
        (ClassTrial, "block"),
        (ClassTrial, "block_constructor"),
    ],
)
def test_timeout_stops_trial_and_continues(
    matrix_chain, tmp_path, constructor, outcome
):
    original = deepcopy(matrix_chain)
    children = {process.pid for process in multiprocessing.active_children()}
    marker = tmp_path / "entered.txt"
    bench = Benchmark(default_solvers=False).add_mdp(matrix_chain)
    bench.add_solver(
        "blocked", solve_function=constructor, outcome=outcome, marker=marker
    )
    bench.add_solver("mdpforge_vi")
    start = time.perf_counter()
    failed, successful = bench.run(0.9, repeats=1, timeout=10)
    assert time.perf_counter() - start < 30
    assert marker.is_file(), "The deadline must allow the solver to start"
    assert failed["status"] == "timeout"
    assert "10 s" in failed["error"]
    for field in ("runtime", "residual", "error_bound"):
        assert np.isnan(failed[field])
    assert successful["status"] == "success", successful["error"]
    assert successful["error_bound"] <= 1e-3
    assert {process.pid for process in multiprocessing.active_children()} == children
    assert_array_equal(matrix_chain.reward_matrix, original.reward_matrix)
    for actual, expected in zip(
        matrix_chain.transition_matrix, original.transition_matrix
    ):
        for field in ("data", "indices", "indptr"):
            assert_array_equal(getattr(actual, field), getattr(expected, field))
    path = bench.export_csv(tmp_path / "timeouts.csv")
    with path.open(newline="", encoding="utf-8") as handle:
        statuses = [row["status"] for row in csv.DictReader(handle)]
    assert statuses == ["timeout", "success"]


@pytest.mark.parametrize("constructor", [function_trial, ClassTrial])
@pytest.mark.parametrize(
    "outcome", ["crash", "exception", "invalid", "imprecise", "mutate"]
)
def test_isolated_failures_are_recorded_and_inputs_preserved(
    matrix_chain, constructor, outcome
):
    original = deepcopy(matrix_chain)
    bench = Benchmark(default_solvers=False).add_mdp(matrix_chain)
    bench.add_solver("broken", solve_function=constructor, outcome=outcome)
    bench.add_solver("mdpforge_vi")
    failed, successful = bench.run(0.9, repeats=1, timeout=10)
    assert failed["status"] == (
        "imprecise" if outcome in {"imprecise", "mutate"} else "error"
    )
    if outcome == "crash":
        assert "exited with code 23" in failed["error"]
        assert np.isnan(failed["runtime"])
    elif outcome == "exception":
        assert "RuntimeError: trial failed" in failed["error"]
        assert np.isfinite(failed["runtime"])
    elif outcome == "invalid":
        assert "finite vector" in failed["error"]
    else:
        # Certification must use the original rewards, including after mutation.
        assert failed["error_bound"] == pytest.approx(20)
    assert successful["status"] == "success", successful["error"]
    assert_array_equal(matrix_chain.reward_matrix, original.reward_matrix)
    for actual, expected in zip(
        matrix_chain.transition_matrix, original.transition_matrix
    ):
        for field in ("data", "indices", "indptr"):
            assert_array_equal(getattr(actual, field), getattr(expected, field))


@pytest.mark.parametrize("constructor", [function_trial, ClassTrial])
def test_spawn_reproduces_paired_seeds_and_restores_parent_rng(
    matrix_chain, constructor
):
    numpy_state, random_state = np.random.get_state(), random.getstate()
    bench = Benchmark(default_solvers=False).add_mdp(matrix_chain)
    bench.add_solver("first", solve_function=constructor, outcome="success")
    bench.add_solver("second", solve_function=constructor, outcome="success")
    direct = bench.run(0.9, precision=1e-4, repeats=2, seed=7)
    isolated = bench.run(0.9, precision=1e-4, repeats=2, seed=7, timeout=10)
    for local, child in zip(direct, isolated):
        for field in (
            "solver",
            "repeat",
            "seed",
            "residual",
            "error_bound",
            "status",
            "error",
        ):
            assert child[field] == local[field]
        assert child["status"] == "success", child["error"]
        assert np.isfinite(child["runtime"]) and child["runtime"] >= 0
    assert isolated[0]["residual"] == isolated[2]["residual"]
    assert isolated[1]["residual"] == isolated[3]["residual"]
    assert isolated[0]["residual"] != isolated[1]["residual"]
    current = np.random.get_state()
    assert current[0] == numpy_state[0]
    assert_array_equal(current[1], numpy_state[1])
    assert current[2:] == numpy_state[2:]
    assert random.getstate() == random_state


def test_large_result_does_not_block_worker_exit():
    model = MDP.from_matrices(
        "large_result", [eye(16384, format="csr")], np.ones((16384, 1))
    )
    bench = Benchmark(default_solvers=False).add_mdp(model)
    bench.add_solver("constant", solve_function=constant_reward_trial)
    row = bench.run(0.9, repeats=1, timeout=10)[0]
    assert row["status"] == "success", row["error"]
    assert row["error_bound"] <= 1e-3


def test_unpickleable_solver_is_reported_and_next_trial_runs(matrix_chain):
    def local_solver(*args):
        pytest.fail("A local solver must not run with spawn")

    bench = Benchmark(default_solvers=False).add_mdp(matrix_chain)
    bench.add_solver("local", solve_function=local_solver)
    bench.add_solver("mdpforge_vi")
    failed, successful = bench.run(0.9, repeats=1, timeout=10)
    assert failed["status"] == "error"
    assert "pickle" in failed["error"].lower()
    assert np.isnan(failed["runtime"])
    assert successful["status"] == "success", successful["error"]


@pytest.mark.parametrize(
    "timeout", [0, -1, np.nan, np.inf, True, False, "1", np.bool_(True)]
)
def test_invalid_timeout_is_rejected(matrix_chain, timeout):
    with pytest.raises(ValueError, match="timeout must be finite and positive"):
        Benchmark().add_mdp(matrix_chain).run(0.9, timeout=timeout)
