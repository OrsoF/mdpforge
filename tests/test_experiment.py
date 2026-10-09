import csv
import json
from copy import deepcopy
from types import SimpleNamespace
from uuid import UUID

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from mdpforge import Benchmark
from mdpforge.solvers.mdpforge_vi import Solver as VI
from mdpforge.utils.experiment import capture_experiment, model_config


@pytest.fixture(autouse=True)
def fixed_provenance(monkeypatch):
    monkeypatch.setattr(
        "mdpforge.utils.experiment._git_metadata",
        lambda: {"commit": "abc123", "dirty": False, "diff_sha256": "0" * 64},
    )
    monkeypatch.setattr(
        "mdpforge.utils.experiment._machine_metadata",
        lambda: {"processor": "test CPU", "logical_cpu_count": 2},
    )


def test_manifest_matches_csv_and_is_a_run_snapshot(chain, tmp_path):
    chain.parameters = {"temperature": np.float64(0.5)}
    initial_value = np.array([1.8, 2.0, 0.0])
    bench = Benchmark(default_solvers=False).add_mdp(chain)
    bench.add_solver("warm VI", solve_function=VI, initial_value=initial_value)
    results = bench.run(0.9, precision=1e-4, repeats=2, seed=7)
    original_manifest = deepcopy(bench.experiment)
    experiment_id = original_manifest["experiment_id"]
    assert str(UUID(experiment_id)) == experiment_id
    assert {row["experiment_id"] for row in results} == {experiment_id}

    # Later changes must not rewrite the configuration that produced these rows.
    chain.parameters["temperature"] = 10
    initial_value[:] = -99
    path = bench.export_csv(tmp_path / "nested" / "results.csv")
    manifest = json.loads(path.with_suffix(".experiment.json").read_text())
    assert manifest["results_file"] == "results.csv"
    assert manifest["experiment_id"] == experiment_id
    assert manifest["schema_version"] == 1
    assert manifest["models"][chain.name]["config"]["parameters"] == {
        "temperature": 0.5
    }
    options = manifest["solvers"]["warm VI"]["options"]
    assert options["initial_value"]["values"] == [1.8, 2.0, 0.0]
    assert bench.experiment == original_manifest
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["experiment_id"] for row in rows] == [experiment_id] * 2
    assert manifest["settings"]["trial_seeds"] == [7, 8]
    assert manifest["environment"]["numpy"] == np.__version__
    assert manifest["code"]["git"]["commit"] == "abc123"


def test_defaults_are_recorded_without_extra_solver_construction(chain):
    calls = []

    class Solver(VI):
        def __init__(
            self, model, discount, final_precision=1e-3, *, budget=10, mode="plain"
        ):
            calls.append((budget, mode))
            super().__init__(model, discount, final_precision)

    bench = Benchmark(default_solvers=False).add_mdp(chain)
    bench.add_solver("configured", solve_function=Solver, budget=4)
    bench.run(0.9, repeats=1)
    assert calls == [(4, "plain")]
    metadata = bench.experiment["solvers"]["configured"]
    assert metadata["options"] == {"budget": 4}
    assert metadata["parameters"] == {"budget": 4, "mode": "plain"}
    assert metadata["source_sha256"] is not None


def test_fingerprints_follow_actual_data_and_preserve_csr(chain):
    def fingerprint(model):
        manifest = capture_experiment([model], {"VI": (VI, {})}, 0.9, 1e-3, 1, 0)
        return manifest["models"][model.name]["data_sha256"]

    original = fingerprint(chain)
    equivalent = deepcopy(chain)
    # Duplicate entries and an explicit zero represent the same transition.
    equivalent.transition_matrix[0] = csr_matrix(
        ([0.5, 0.5, 0.0, 1.0, 1.0], [1, 1, 2, 2, 2], [0, 3, 4, 5]),
        shape=(3, 3),
    )
    before = equivalent.transition_matrix[0].data.copy()
    assert fingerprint(equivalent) == original
    np.testing.assert_array_equal(equivalent.transition_matrix[0].data, before)
    changed_rewards = deepcopy(chain)
    changed_rewards.reward_matrix[0, 0] += 1
    assert fingerprint(changed_rewards) != original
    changed_transitions = deepcopy(chain)
    changed_transitions.transition_matrix[0] = csr_matrix(np.eye(3))
    assert fingerprint(changed_transitions) != original


def test_external_and_custom_model_configurations(chain):
    external = SimpleNamespace(**vars(chain), scale=np.int64(2), _private="hidden")
    config = model_config(external)
    assert config["scale"] == 2
    assert "transition_matrix" not in config
    assert "reward_matrix" not in config
    assert "_private" not in config

    external.get_config = lambda: {"recipe": {"scale": np.float64(0.25)}}
    manifest = capture_experiment([external], {"VI": (VI, {})}, 0.9, 1e-3, 1, 0)
    assert manifest["models"][chain.name]["config"] == {"recipe": {"scale": 0.25}}

    class SlottedModel:
        __slots__ = (
            "state_dim",
            "action_dim",
            "name",
            "transition_matrix",
            "reward_matrix",
            "scale",
        )

    slotted = SlottedModel()
    for name in SlottedModel.__slots__:
        setattr(slotted, name, getattr(external, name))
    assert model_config(slotted)["scale"] == 2


def test_ids_change_between_runs_and_registration_invalidates_manifest(chain):
    bench = Benchmark().add_mdp(chain)
    first = bench.run(0.9, repeats=1)[0]["experiment_id"]
    second = bench.run(0.9, repeats=1)[0]["experiment_id"]
    assert first != second
    bench.add_solver("extra VI", solve_function=VI)
    assert bench.experiment is None
    assert bench.results == []


def test_failed_trials_export_strict_json_with_unsupported_options(chain, tmp_path):
    def broken_solver(
        transitions, rewards, discount, precision, *, token, limit=np.inf
    ):
        raise RuntimeError("deliberately broken")

    bench = Benchmark(default_solvers=False).add_mdp(chain)
    bench.add_solver("broken", solve_function=broken_solver, token=object())
    assert bench.run(0.9, repeats=1)[0]["status"] == "error"
    path = bench.export_csv(tmp_path / "failed.csv")

    def reject_non_json_constant(value):
        raise AssertionError(f"Non-JSON constant: {value}")

    manifest = json.loads(
        path.with_suffix(".experiment.json").read_text(),
        parse_constant=reject_non_json_constant,
    )
    parameters = manifest["solvers"]["broken"]["parameters"]
    assert parameters["limit"] == {"nonfinite_float": "inf"}
    assert parameters["token"] == {"unserialized_type": "builtins.object"}


def test_manifest_records_timeout_execution_mode(chain):
    manifest = capture_experiment(
        [chain], {"VI": (VI, {})}, 0.9, 1e-3, 1, 0, timeout=np.float64(0.5)
    )
    assert manifest["settings"]["timeout"] == 0.5
    assert manifest["settings"]["execution_mode"] == "spawn"


def test_function_defaults_use_calling_convention_not_parameter_names(chain):
    def solver(p, r, gamma, epsilon, *, env="example"):
        return np.array([2 * gamma, 2, 0])

    bench = Benchmark(default_solvers=False).add_mdp(chain)
    bench.add_solver("function", solve_function=solver)
    bench.run(0.9, repeats=1)
    assert bench.experiment["solvers"]["function"]["parameters"] == {"env": "example"}
