# mdpforge tests

Keep model behavior, persistence, solver contracts and scientific validation distinct. Tests must not duplicate the algorithms they check.

| File | Responsibility |
| --- | --- |
| [`test_mdp.py`](test_mdp.py) | Construction, input conversion, CSR formats, densities and state relabeling. |
| [`test_persistence.py`](test_persistence.py) | Model cache round trips and optimal-value cache reuse by name and discount. |
| [`test_model_validation.py`](test_model_validation.py) | Rejection of structurally invalid or nonfinite models. |
| [`test_solver_contracts.py`](test_solver_contracts.py) | Solver input preservation and compatibility with plain model objects. |
| [`test_solvers_rooms.py`](test_solvers_rooms.py) | Scientific validation of installed catalogue solvers on the 10,000-state Rooms small preset. |

## Adding a solver

[`conftest.py`](conftest.py) provides fixtures and configures solver contract cases. Some backends and model/discount combinations are excluded from those cases; the fixture definitions show the exact selection. The Rooms test independently includes every discovered solver.

Tests use constructor options and argument-free `run()` calls. They exercise public behavior without calling solver internals or fixing backend computation steps and counters. Adapter tests may check arguments passed to public backend APIs.

The Rooms test calls `validate_result()` with discount 0.8 and precision 1e-3 to certify each result using its declared validation method. Contract tests separately check input preservation on small synthetic instances.

[`_contracts.py`](_contracts.py) provides synthetic models and input-preservation checks. [`_solver_support.py`](_solver_support.py) discovers solvers without importing optional backends and runs external cases in isolated processes when their dependencies are installed; missing dependencies are skipped.

When tests are explicitly requested, add a specific regression only for behavior beyond the common interface: a pathological case, external conversion, requested numerical precision, partitioning, caching, timeout or export. Keep concrete counterexamples that prevent real regressions from returning.

## Commands

Use Python 3.11+ in Conda `benchmark`, with the project and development dependencies installed. **Run tests only when explicitly requested.** Run the full suite, including installed backends, without a separate backend mode:

```text
conda run -n benchmark python -m pytest
```

To list tests without executing them, use `conda run -n benchmark python -m pytest --collect-only -q` when requested. See [AGENTS.md](../AGENTS.md) for development and reporting rules.
