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
import numpy as np
from mdpforge import Benchmark

P = [[[1, 0], [0, 1]], [[0, 1], [1, 0]]]
R = [[1, 0], [0, 2]]

def my_solver(transitions, rewards, discount, precision):
    value = np.zeros(rewards.shape[0])
    while True:
        updated = np.max([
            rewards[:, a] + discount * (p @ value)
            for a, p in enumerate(transitions)
        ], axis=0)
        if np.max(np.abs(updated - value)) <= precision * (1 - discount):
            return updated
        value = updated

bench = Benchmark(default_solvers=False)
bench.add_mdp("rooms")                               # Existing MDP
bench.add_mdp("my_mdp", transitions=P, reward=R)      # New MDP
bench.add_solver("personal_mpi")                     # Existing solver
bench.add_solver("my_solver", solve_function=my_solver)  # New solver

results = bench.run(discount=0.9, precision=1e-3)
bench.export_csv("results.csv")
```

Every pair runs three trials at the same target precision. Results include
`runtime`, `error_bound` and `status`; compare successful trials.
`Benchmark()` includes VI by default.

## Define an MDP

- `transitions[action]`: an `(S, S)` matrix, dense or sparse.
- `reward[state, action]`: expected immediate rewards, shape `(S, A)`.

Dimensions, CSR conversion and validation are automatic. Use a unique custom name;
[catalogue models](src/mdpforge/models) are selected by their module name.

## Define a solver

Provide `solve_function(transitions, rewards, discount, precision)` returning a
NumPy value vector of length `S`. Inputs are CSR transitions and NumPy rewards.
The benchmark handles copies, timing and final precision verification.

`solver_name` labels the results and must be unique. Without `solve_function`, it
selects a [catalogue solver](src/mdpforge/solvers). Extra options are forwarded as
keyword arguments. Solver classes are also accepted through `solve_function`.

[MIT license](LICENSE).
