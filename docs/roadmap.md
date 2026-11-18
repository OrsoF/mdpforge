# Roadmap

Work through priorities 1, 2 and 3 in order. Complete priority 4 before the broad comparison campaign.

## 1. Benchmark result-validation coverage

- [ ] Add small deterministic tests for `Benchmark` and `ScalingBenchmark` covering validation disabled, certification passed, an out-of-precision result and an exception during validation.
- [ ] Verify certification runs after solver timing, in the parent process and outside the solver timeout. Do not add reference solves or repeat solver execution.
- [ ] Verify certification failure preserves solver execution status and runtime. Check separate validation status, errors and runtime, with unavailable residuals and error bounds kept as `NaN`.
- [ ] Verify validation diagnostics survive CSV export and reload, and the experiment manifest records validation mode.
- [ ] Cover the total-reward certificate's restriction to absorbing acyclic models and rejection of validation for average reward.

Completion: tests protect the separation between solver execution, measured runtime and certified quality, including cases where quality is unchecked.

## 2. Requested precision and certified quality

- [ ] Map each solver's public precision option, the value passed to its backend, the stopping threshold's meaning, the returned result and the available certificate, separately for each reward criterion.
- [ ] Start with the discounted solvers selected for the MPI study. Distinguish value error, policy quality and certified bounds; equal numerical tolerances do not establish equal solution quality.
- [ ] Identify where the requested final target can be certified, where the available bound is too loose to establish it and where no suitable certificate exists. Keep these cases explicit in comparisons, separate from execution success.
- [ ] Record the current limits for total reward and the absence of average-reward certification. Treat any proposed correction to public backend parameters as a separate change; preserve existing solver computations and internal stopping criteria.

Completion: every selected solver has an explicit mapping to the common quality target or a stated limit on comparability.

## 3. MPI inner-precision study

- [ ] Select a few contrasting model families and discounted MPI implementations exposing an inner tolerance; record each option's meaning and scaling. The current total-reward `mdpforge_mpi` exposes an inner iteration budget but no `evaluation_precision`, so study its budget separately.
- [ ] Sweep `evaluation_precision` on a logarithmic grid, including the default, at fixed discount, final precision and inner iteration budget; keep model instances, paired seeds, thread counts and other solver settings fixed.
- [ ] Repeat these sweeps for selected discounts, final precision targets and inner iteration budgets, changing one setting between sweeps to identify interactions rather than mixing their effects.
- [ ] Compare runtime over the planned repeats, independently certified final quality, failures and timeouts. Distinguish inner threshold attainment from iteration-budget exhaustion where observable; report unknown stopping reasons as unknown. Plan any missing diagnostics separately while preserving existing solver internals and timing boundaries.
- [ ] Report when extra inner work reduces total runtime, when tighter thresholds have no effect because the budget binds, and whether a useful default depends on discount, final precision or model family. Treat identical outputs or runtimes alone as insufficient evidence that the budget binds.
- [ ] Save a CSV and matching experiment manifest for each configuration, including validation mode; distinguish campaigns with and without validation.

Completion: the study identifies when tighter inner thresholds help, add unnecessary cost or have no explained effect, with evidence for each conclusion.

## 4. Catalogue balance and representative comparison

- [ ] Audit family definitions, task categories, parameter and size conventions, unavailable sizes, duplicate or closely related instances, and overrepresented families in benchmark selections and rankings.
- [ ] Compare actual state counts, action counts and transition nonzeros; matching size labels do not establish comparable difficulty.
- [ ] Define a panel that prevents families with many instances from dominating rankings by their count alone. Fix instances and selection rules before the campaign, with a reason for each selected instance.
- [ ] Present results by family and across the full panel, with repeat coverage and certified versus unchecked results explicit. Distinguish campaigns with and without validation.
- [ ] Regenerate comparison figures with current backend versions and complete settings. Keep a CSV and matching manifest for each configuration, and require all expected repeats to succeed for runtime-comparison eligibility. Follow the [visualization roadmap](visualization-roadmap.md) for figure interpretation and README additions.

Completion: the panel has explicit selection rules, rankings report their coverage and every figure can be traced to its data and settings.

## ScalingBenchmark extensions

The [scaling notebook](../notebooks/benchmark_scaling.ipynb) provides a small study using the current API. Implement quality summaries and dispersion plots first, then additional sweep parameters. Plan progressive execution separately when campaign memory becomes limiting, and cross-condition comparisons when studying interactions.

- [ ] Extend `scaling_summary()` with per-size counts of fully certified instances, certification failures and unchecked or incomplete cases. Add an explicit certified-only comparison option requiring every expected repeat to execute successfully and pass certification; keep all planned instances in coverage counts and pair speedups on the same eligible instances.
- [ ] Plot the existing runtime quartiles across eligible instance medians, with optional instance points. Distinguish variation between model instances from repeat variation; report the contributing counts and do not present quartile bands as confidence intervals.
- [ ] Support controlled sweeps over action count or successors per state, with the varied parameter, requested and actual values, and fixed settings recorded in the manifest. Update grouping as well as axis labels; current `x="nnz"` plots remain grouped by actual state count. Keep existing state sweeps and seed behavior unchanged by default.
- [ ] Design an explicit progressive execution mode that constructs one instance, runs its trials, saves results and releases its matrices before constructing the next. Record execution order and mode, preserve timing boundaries and paired seeds, and avoid retaining all model matrices or solutions. Do not introduce implicit warm-ups, extra repeats or cache clearing.
- [ ] Compare separate scaling runs at different discounts or final precision targets. Keep a CSV and manifest per condition; verify matching model instances, solver options except deliberately varied parameters, repeat plans, thread settings and validation mode before overlaying curves. Preserve distinct conditions rather than pooling their summaries.

Completion: scaling results show execution and certification coverage, instance dispersion and the actual varied parameter. Larger campaigns can opt into progressive execution, and cross-condition plots identify their controlled differences. See the [requirements](requirements.md#scaling-studies) for the study contract.

## Interfaces and benchmarks

- [ ] Finish migrating solvers and callers to `SolveResult`; remove solver result attributes `value`, `policy` and `runtime`.
- [ ] Simplify `Benchmark` by moving helpers into utilities or separating responsibilities.
- [ ] Decide which parameters belong in `METADATA` or model/solver constructors, and whether solvers need a shared options format and final precision target.

## Tests and backends

- [ ] Define model-specific test coverage beyond the shared contract, including arguments and useful edge cases.
- [ ] Diagnose and resolve madupite solver failures.

## Algorithms and experiments

- [ ] Review VI variants to add and explain the C++/Python VIGS performance gap.
- [ ] Decide whether VISOR needs additional registered configurations beyond its constructor options.
- [ ] Select other solvers to add.
- [ ] Define a method for tuning solver parameters.
- [ ] Compare MDP formulations as constrained linear programs against standard MDP methods, identifying the primal or dual formulation and optimization backend.
- [ ] Explain solver performance through model structure: action counts, transition nonzeros, cycles and terminal reachability at comparable sizes. Define the experimental route later.
- [ ] Compare aggregated methods with their non-aggregated baselines using runtime, solution quality and memory use to identify when aggregation pays off and whether it saves representation space. Distinguish input storage, aggregate representation and temporary storage from peak process memory; define memory profiling separately from runtime measurements.
- [ ] Plan experiments against standard methods, identify feasible improvements and demonstrate a scientific contribution. See the [visualization roadmap](visualization-roadmap.md) for measurements and README additions.
