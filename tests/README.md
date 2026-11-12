# mdpforge tests

Tests cover model behavior, persistence, solver interfaces and solution quality without duplicating solver algorithms.

| File | Responsibility |
| --- | --- |
| [`test_mdp.py`](test_mdp.py) | Construction, input conversion, CSR formats, densities and state relabeling. |
| [`test_persistence.py`](test_persistence.py) | Model cache round trips and optimal-value cache reuse by name and discount. |
| [`test_model_validation.py`](test_model_validation.py) | Rejection of structurally invalid or nonfinite models. |
| [`test_solver_contracts.py`](test_solver_contracts.py) | Solver input preservation and compatibility with plain model objects. |
| [`test_solvers_rooms.py`](test_solvers_rooms.py) | Discounted validation on 100-state Rooms at discount 0.8 and precision 1e-3. |
| [`test_total_solvers_rooms.py`](test_total_solvers_rooms.py) | Total-reward values against the exact solution on cyclic 100-state Rooms. |
| [`test_average_solvers_rooms.py`](test_average_solvers_rooms.py) | Finite values and exit-reaching policies on 100-state Rooms. |

## Solver coverage

[`conftest.py`](conftest.py) selects discounted contract cases, with explicit exclusions. The three Rooms modules independently include every discovered solver for their criterion. Average-reward checks are specific to Rooms; the library has no general average-reward certificate.

[`_contracts.py`](_contracts.py) supplies synthetic models, reproducible RNG contexts and input-preservation checks. [`_solver_support.py`](_solver_support.py) discovers optional backends and isolates external cases in subprocesses; missing dependencies are skipped.

Use constructor options and argument-free `run()` calls. Check public behavior, input preservation and meaningful numerical cases; adapter tests may inspect arguments passed to public backend APIs. Add specific regressions for behavior beyond the common contract.

## Commands

**Add tests or run pytest/Ruff only when explicitly requested.** Use Python 3.11+ in Conda `benchmark` and run the full suite, including all installed backends:

```text
conda run -n benchmark python -m pytest
```

See [AGENTS.md](../AGENTS.md) for setup and code-quality commands.
