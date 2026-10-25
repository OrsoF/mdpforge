# mdpforge tests

Keep **shared contracts**, **scientific properties** and **specific regressions** distinct. Tests must not duplicate the algorithms they check.

## Adding a solver

[`conftest.py`](conftest.py) discovers solver modules and configures the cases used by [`test_workflows.py`](test_workflows.py). The current workflow checks input preservation and compatibility with a plain object exposing model data. Some backends and model/discount combinations are currently excluded; the fixture definitions show the exact coverage.

Tests use constructor options and argument-free `run()` calls. They exercise public behavior without calling solver internals or fixing backend computation steps and counters. Adapter tests may check arguments passed to public backend APIs.

The previous comparisons with `V*`, Bellman-residual bounds, policy-quality checks and output-reproducibility checks have been removed. A new shared value-function check is planned in the [roadmap](../docs/roadmap.md). The sparse-model tests still check that policy solvers preserve CSR transitions without densifying them.

Optional native backends run in isolated processes when their dependencies are installed; missing dependencies are skipped. [`_contracts.py`](_contracts.py) provides shared fixtures and input checks, while [`_fake_backends.py`](_fake_backends.py) provides controlled backend doubles. These are helpers, not standalone tests.

When tests are explicitly requested, add a specific regression only for behavior beyond the common interface: a pathological case, external conversion, requested numerical precision, partitioning, caching, timeout or export. Keep concrete counterexamples that prevent real regressions from returning.

## Commands

Use Python 3.11+ in Conda `benchmark`, with the project and development dependencies installed. **Run tests only when explicitly requested.** Run the full suite, including installed backends, without a separate backend mode:

```text
conda run -n benchmark python -m pytest
```

To list tests without executing them, use `conda run -n benchmark python -m pytest --collect-only -q` when requested. See [AGENTS.md](../AGENTS.md) for development and reporting rules.
