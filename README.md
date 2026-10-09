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
from mdpforge import Benchmark

P = [[[1, 0], [0, 1]], [[0, 1], [1, 0]]]  # transitions[action][state][next_state]
R = [[1, 0], [0, 2]]                       # rewards[state][action]

bench = Benchmark()
bench.add_mdp("rooms")
bench.add_mdp("my_mdp", transitions=P, reward=R)
results = bench.run(discount=0.9)
```

Select a [catalogue model](src/mdpforge/models) by module name; its `Model()` chooses
the dimensions and `create_model()` builds or loads it before solver timing.
Unknown names require `transitions` and `reward` (`rewards` is also accepted);
dimensions, CSR conversion and validation are automatic. Catalogue names cannot
be combined with matrices. Built instances are accepted with `add_mdp(model)`.
VI and QVI run by default, comparing times at the same final precision
(`1e-3`, three trials). Each result includes `solver`, `runtime`, `error_bound` and
`status`; compare successful trials. Adjust with `run(discount=0.9, precision=1e-4)`.

To save or plot results, use `bench.export_csv("results.csv")` or
`bench.plot_heat(reference="VI")` (requires the `plot` extra).

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

In [solvers](src/mdpforge/solvers), inherit from
[`GenericSolver`](src/mdpforge/core/solver.py) and implement only the algorithm:

```python
from mdpforge.core.solver import GenericSolver

class Solver(GenericSolver):
    def _solve(self):
        return my_algorithm(self.model, self.discount, self.final_precision)

bench.add_solver(Solver, name="my_solver")
```

Replace `my_algorithm` with your algorithm, returning one value per state.
It runs alongside VI and QVI; use `Benchmark(default_solvers=False)` to select
only your own solvers.
The base provides the constructor, `run()`, NumPy `value`, timing and
`policy = None`; set `self.policy` if your algorithm computes one. The benchmark
checks the final precision. For custom options, define a constructor calling
`super().__init__(model, discount, final_precision)` and register with
`bench.add_solver(Solver, **options)`.

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
