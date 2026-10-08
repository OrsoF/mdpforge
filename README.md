# mdpforge

Research code for finite Markov decision processes with NumPy and SciPy.
The goal is to compare new MDP models and solvers through small, reproducible
experiments. The project is still under development.

The current examples include Rooms, Schoolboy, and RiverSwim with
value iteration (VI) and modified policy iteration (MPI).

## Installation

Python 3.11 or newer. From the repository root:

```sh
python -m pip install -e ".[dev]"
```

Use `python -m pip install -e .` for the core library only.
Dependencies are declared in [pyproject.toml](pyproject.toml).

## Example

```python
from mdpforge.models.rooms import Model
from mdpforge.solvers.personal_vi import Solver

model = Model(100, 4)
model.create_model(save=False)
solver = Solver(model, discount=0.9)
solver.run()
print(solver.value)
```

## Development checks

Tests discover modules directly in `models/` and `solvers/`, using
`Model(state_dim, action_dim)` and `Solver(model, discount, final_precision=...)`.
Specialized subpackages require dedicated tests.
Models can declare `TEST_PARAMETERS` for a small valid instance (default: 7 states,
4 actions). Integrations beyond NumPy/SciPy require `pytest --optional`; they run
in isolated processes with a 30-second timeout, and missing dependencies are skipped.
MDPSolver uses serial execution by default; `parallel=True` opts in to parallelism.

```sh
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

For the project's Conda environment, prefix these commands with
`conda run -n benchmark`.

[MIT license](LICENSE).
