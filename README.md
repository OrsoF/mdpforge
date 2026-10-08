# mdpforge

Benchmark finite Markov decision processes: compare a new solver on existing
MDPs, or compare existing solvers on a new MDP. Built with NumPy and SciPy.
The project is experimental and supports discounted problems (`0 < discount < 1`).

## Install

Python 3.11 or newer. From the repository root:

```sh
python -m pip install -e .
```

## Run a benchmark

This example compares VI and QVI on Rooms (100 states, 4 actions), with three
trials per solver and a requested value error of at most `1e-3`.

```python
from mdpforge.core.benchmark import benchmark, export_csv
from mdpforge.models.rooms import Model
from mdpforge.solvers.personal_vi import Solver as VI
from mdpforge.solvers.personal_qvi import Solver as QVI

model = Model(100, 4)
model.create_model(save=False)

results = benchmark(
    model,
    {"VI": VI, "QVI": QVI},
    discount=0.9,
    epsilon=1e-3,
    repeats=3,
    seed=0,
)
for row in results:
    print(row["solver"], row["status"], row["runtime"], row["error_bound"])
export_csv(results, "artifacts/tmp/benchmark.csv")
```

`benchmark()` constructs and runs each solver on a fresh copy of the built model.
It returns one dictionary per trial; `export_csv()` writes the same measurements.

## Read the results

| Field | Meaning |
| --- | --- |
| `runtime` | Seconds spent constructing the solver and executing `run()`. |
| `residual` | Maximum absolute difference between the final value and one optimal Bellman update. |
| `error_bound` | Upper bound on the maximum difference from the optimal value: `residual / (1 - discount)`. |
| `status` | `success`: bound ≤ epsilon; `imprecise`: bound exceeds epsilon; `error`: exception or invalid value. |
| `error` | Explanation when the trial fails. |

Compare median runtimes of successful trials, keeping the model, discount and
epsilon fixed. Model copying and independent verification are excluded from timing.
VI uses `final_precision=epsilon`; PI/MPI tolerances are algorithm-specific, but
the benchmark checks every solver against the same final error bound.

## Add a model or solver

- **Model:** add `Model(GenericModel)` in [models](src/mdpforge/models).
  Implement `_build_model()` and call `create_model()`. Expose `state_dim`, `action_dim`, transitions
  `transition_matrix[action]` of shape `(states, states)`, and rewards
  `reward_matrix` of shape `(states, actions)`.
- **Solver:** add a `Solver` class in [solvers](src/mdpforge/solvers). Its constructor
  takes `(model, discount, final_precision=...)`; `run()` sets `value`, a vector of
  length `state_dim`. Repository tests also expect `policy` (or `None`) and `runtime`.
  Add the constructor to the dictionary passed to `benchmark()`.

Transitions have one format: a list of SciPy `csr_matrix` objects, finalized by
`create_model()` after building or loading. Rewards and values remain NumPy arrays.
Solvers use this format directly; external backend objects are built separately.

## Development

```sh
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

In the project's Conda environment, prefix commands with `conda run -n benchmark`.
Tests discover modules in `models/` and `solvers/`; models can provide
`TEST_PARAMETERS` for a small valid instance.

External backends require their [optional dependencies](pyproject.toml), e.g.
`python -m pip install -e ".[mdptoolbox]"`. Run their tests with `python -m pytest --optional`;
missing dependencies are skipped. In the pinned MDPToolbox version, VI can fail
with constant optimal immediate rewards; Gauss-Seidel fails on CSR inputs under
NumPy 2 because its native loop converts a one-element array to a scalar.
MDPSolver runs serially by default.

[MIT license](LICENSE).
