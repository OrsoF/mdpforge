# Current requirements

## Purpose

mdpforge is a research tool for finite Markov decision processes (MDPs). Its current use is to answer four questions through reproducible benchmarks:

1. **Industrial solver audit:** which established solver and backend performs best overall across a representative model set, considering runtime, solution quality, failures and timeouts?
2. **Linear solver competitiveness:** are linear solvers competitive with standard MDP methods, and on which model sizes and structures? Identify the formulation and backend used for each comparison.
3. **MPI inner precision:** how does the precision of the inner policy-evaluation loop in modified policy iteration (MPI) affect total runtime and final solution quality, and which value performs best overall?
4. **Aggregated methods:** in [JAIR.ipynb](../notebooks/JAIR.ipynb), how do `mdpforge_aggvi`, `mdpforge_aggqvi` and `mdpforge_aggmpi` compare with each other and with the notebook's VI, QVI, MPI and TVI baselines on total-reward models? Measure runtime, solution quality, failures and timeouts under the same benchmark settings.

## Intended workflow

1. Select catalogue models and solvers, or register custom ones through the Python API.
2. Set the reward criterion, model parameters, solver options, precision, seeds, repeats and trial timeout explicitly.
3. Run comparisons with `Benchmark`; use `ScalingBenchmark` to study model families at several sizes.
4. Request solution certification with `validate=True` when supported, separately from solver timing.
5. Export results and experiment settings, reload them, and compare runtime, failures, coverage and certified error bounds through summaries and plots.

## Requirements

- **Easy extension:** new models and solvers must fit the shared interfaces without changing benchmark orchestration. Models provide sparse transition matrices and expected rewards; solvers accept constructor options and expose an option-free `run()`.
- **Fair comparisons:** compare solvers on identical model instances with paired trial seeds and explicit precision targets. Runtime includes solver construction, backend conversions and execution; model preparation, validation and exports stay outside timing.
- **Reproducible evidence:** retain model identities and parameters, solver settings, seeds, versions, machine information, validation mode and trial outcomes in CSV exports and their experiment manifests.
- **Honest results:** distinguish execution success from numerical accuracy. Record errors and timeouts, keep validation outcomes separate, and leave unavailable diagnostics as `NaN`. Runtime rankings require all expected repeats to succeed.
- **Controlled experiments:** audit industrial and linear solvers on the same cases and final accuracy targets. For MPI, vary `evaluation_precision` while keeping the final precision, inner iteration budget and other settings fixed; report when the budget limits evaluation. Explain cold versus warm measurements and support performance claims with saved results and solution-quality checks.
- **Scientific stability:** preserve existing solver algorithms, numerical conventions, reward scales, random behavior and reference caches. Keep optional backends separate from core dependencies.

## Scope and completion criteria

Support discounted, finite total and average reward. Discounted certification is available; total certification currently covers absorbing acyclic models; average certification is unavailable. Define the audited solver versions, model set, parameter ranges and aggregation rule before each study; report reward criteria separately.

The studies must provide an overall solver ranking with coverage and performance profiles, a verdict on linear solver competitiveness, an MPI inner-precision sweep with a recommended default, and a comparison of aggregated methods and baselines in `JAIR.ipynb`. State the model set and accuracy target behind each conclusion, show important exceptions, and report when no single solver or precision value is consistently best. Save reloadable results and experiment settings from a short Python script or notebook. See the [model](adding-a-model.md) and [solver](adding-a-solver.md) guides for extension details.
