# Current requirements

## Purpose

mdpforge is a research tool for finite Markov decision processes (MDPs). Its current use is to answer five questions through reproducible benchmarks:

1. **Industrial solver audit:** compare established solvers and backends for total, discounted and average reward. Which solver performs best overall within each criterion across a representative model set, considering runtime, solution quality where assessable, failures and timeouts? Report unsupported solver/criterion combinations explicitly.
2. **Linear programming competitiveness:** is solving an MDP as a constrained linear optimization problem competitive with standard MDP methods, and on which model sizes and structures? Identify the primal or dual formulation and optimization backend used for each comparison.
3. **MPI inner precision:** how does the precision of the inner policy-evaluation loop in modified policy iteration (MPI) affect total runtime and final solution quality, and which value performs best overall?
4. **Aggregated methods:** in [JAIR.ipynb](../notebooks/JAIR.ipynb), how do `mdpforge_aggvi`, `mdpforge_aggqvi` and `mdpforge_aggmpi` compare with each other and with the notebook's VI, QVI, MPI and TVI baselines on total-reward models? Measure runtime, solution quality, failures, timeouts and memory use to establish whether aggregation saves representation space; distinguish representation storage from peak process memory and define memory profiling separately from runtime measurements.
5. **Scaling:** how do runtime, execution coverage and certified quality change with model size or structure, and how do these curves depend on discount or final precision? Use the [scaling notebook](../notebooks/benchmark_scaling.ipynb) for a small study with the current API; the extensions below are requirements, not claims of implemented features.

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
- **Controlled experiments:** audit industrial and linear programming solvers on the same cases and final accuracy targets. For MPI implementations exposing an inner tolerance, vary `evaluation_precision` while keeping the final precision, inner iteration budget and other settings fixed; repeat the sweep at selected discounts, final precision targets and budgets to study interactions. Report observed budget limits and unavailable diagnostics separately. Explain cold versus warm measurements and support performance claims with saved results and solution-quality checks.
- **Scientific stability:** preserve existing solver algorithms, numerical conventions, reward scales, random behavior and reference caches. Keep optional backends separate from core dependencies.

## Scaling studies

- **Controlled parameter:** vary one model parameter per sweep and record requested and actual values, state and action counts, transition nonzeros, model-generation seeds and all fixed settings. Support state count first, then action count and successors per state. Changing an axis to transition nonzeros alone does not constitute a controlled nonzero-count sweep.
- **Instance and repeat variation:** record independent model instances separately from repeated solver trials. Use the median over all expected successful repeats for each eligible instance, then summarize across instances with medians and quartiles. Show contributing instance counts, retain failures and timeouts in coverage, and identify quartile bands as instance dispersion rather than confidence intervals.
- **Certified comparisons:** report fully certified instances, certification failures and unchecked or incomplete cases separately from execution coverage. A certified-only option must require every expected repeat to execute successfully and pass certification, retain the full planned instance denominator and use matched eligible instances for paired speedups. Certification stays outside solver timing; unavailable certificates remain explicit.
- **Large campaigns:** provide an explicit progressive mode that builds, runs, saves and releases one instance at a time, without retaining all matrices or solution vectors. Record its execution order and mode because allocation and cache behavior can differ from constructing all instances first. Preserve trial timing, paired seeds and default cache behavior; memory profiling remains a separate experiment.
- **Cross-condition curves:** keep separate exports and manifests for each discount or final precision target. Before comparison, verify identical model instances, the intended solver configurations, repeat plans, thread settings and validation mode; record every deliberately varied setting. Do not pool distinct conditions into a single runtime summary.

Completion requires reloadable data and manifests, curves with execution and certification coverage, visible instance dispersion, explicit controlled parameters and stated limits on quality comparability. The current notebook demonstrates available size sweeps, saved results and plots; generalized sweeps, library-level certification filters, progressive execution and cross-run compatibility checks remain roadmap work.

## Scope and completion criteria

Support discounted, finite total and average reward. Discounted certification is available; total certification currently covers absorbing acyclic models; average certification is unavailable. Define the audited solver versions, model set, parameter ranges and aggregation rule before each study; report reward criteria separately.

The studies must provide separate industrial solver rankings with coverage and performance profiles for total, discounted and average reward, a verdict on linear programming competitiveness, an MPI inner-precision study with a recommended default where supported by the results, and a comparison of aggregated methods and baselines in `JAIR.ipynb` including memory use and representation savings. State the model set and accuracy target behind each conclusion, show important exceptions, and report when no single solver or precision value is consistently best. Save reloadable results and experiment settings from a short Python script or notebook. See the [model](adding-a-model.md) and [solver](adding-a-solver.md) guides for extension details.
