# mdpforge

Compare solvers on finite Markov decision processes, or benchmark a new MDP.
Experimental, built with NumPy/SciPy; discounted problems only (`0 < discount < 1`).

## Install

Python 3.11+. From the repository root:

```sh
python -m pip install -e .
```

Optional [extras](pyproject.toml): `plot`, `mdptoolbox`, `mdpsolver`, `gurobi`,
`marmote`, `maze`. Example: `python -m pip install -e ".[mdptoolbox]"`.
MDPToolbox uses the official `pymdptoolbox==4.0b3` package.

## Run a benchmark

```python
from mdpforge.core.benchmark import Benchmark
from mdpforge.models.rooms import Model as Rooms
from mdpforge.solvers.personal_vi import Solver as VI
from mdpforge.solvers.personal_qvi import Solver as QVI

model = Rooms()
model.create_model(save=False)

bench = Benchmark().add_mdp(model)
bench.add_solver(VI, name="VI").add_solver(QVI, name="QVI")
results = bench.run(discount=0.9, precision=1e-3, repeats=3, seed=0)
bench.export_csv("artifacts/tmp/benchmark.csv")
```

Register more built models with `add_mdp()`; each must have a unique name.
Every model/solver pair runs on fresh copies, returning one dictionary per trial.
Models have modest defaults; pass dimensions such as `Rooms(100, 4)` to override
them. Some environments determine their actual dimensions internally.
With the `plot` extra, `bench.plot_heat(reference="VI")` compares median runtimes.

| Field | Meaning |
| --- | --- |
| `runtime` | Seconds for solver construction and `run()`; copying and verification excluded. |
| `residual` | Maximum absolute difference between the final value and its optimal Bellman update. |
| `error_bound` | Certified maximum value error: `residual / (1 - discount)`. |
| `status` | `success`: bound ≤ precision; `imprecise`: bound too large; `error`: failed trial, explained in `error`. |

Compare successful trials at fixed model, discount and precision. All solvers
face the same final error bound; heatmaps mark `FAIL` if any repeat misses it.

## Define an MDP

[`MDP`](src/mdpforge/core/mdp.py) supports both generators and existing matrices:

```python
from mdpforge.core.mdp import MDP

model = MDP.from_matrices("my_mdp", transition_matrix, reward_matrix)
```

This infers dimensions, converts transitions to CSR and validates the data;
the model is ready for `add_mdp()`. For a generator, add `Model(MDP)` in
[models](src/mdpforge/models), implement `_build_model()` and call `create_model()`.

Data contract: `name`, `state_dim`, `action_dim`, a list of CSR matrices
`transition_matrix[action]` of shape `(S, S)`, and NumPy `reward_matrix` of shape
`(S, A)`. External objects implementing [`MDPProtocol`](src/mdpforge/core/model.py)
are accepted without inheritance.

`MDP` also provides validation, state relabeling, densities and cached optimal
values. Caches live under `artifacts/`; change `name` when changing model parameters.

## Add a solver

Add `Solver` in [solvers](src/mdpforge/solvers). Its constructor accepts
`(model, discount, final_precision=...)` plus explicit options; `run()` takes no
options and sets NumPy `value` of length `state_dim`, `policy` (or `None`) and
`runtime`. Register it with `bench.add_solver(Solver, name="label", **options)`.

## Development

```sh
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

In Conda, prefix commands with `conda run -n benchmark`. Tests discover models and
solvers, run installed backends and skip missing dependencies. Models may define
`TEST_PARAMETERS` for small instances.

[MIT license](LICENSE).
