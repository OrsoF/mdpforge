# Roadmap

## Interfaces and benchmarks

- [ ] Finish migrating solvers and callers to `SolveResult`; remove solver result attributes `value`, `policy` and `runtime`.
- [ ] Simplify `Benchmark` by moving helpers into utilities or separating responsibilities.
- [ ] Decide which parameters belong in `METADATA` or model/solver constructors, and whether solvers need a shared options format and final precision target.

## Tests and backends

- [ ] Test the benchmark's optional output-precision validation.
- [ ] Define model-specific test coverage beyond the shared contract, including arguments and useful edge cases.
- [ ] Diagnose and resolve madupite solver failures.

## Models and numerical quality

- [ ] Audit the consistency and balance of the full model catalogue: check family definitions, parameter and size conventions, duplicate or closely related instances, and overrepresented families in benchmark selections and rankings.
- [ ] Check that final precision is actually verified outside solver timing, with explicit validation coverage and unchecked cases reported separately from execution success.
- [ ] Audit whether final precision means comparable solution quality across solvers within each reward criterion; map backend tolerances and stopping rules to the common quality target, and state where comparability cannot be established with current certificates.

## Algorithms and experiments

- [ ] Review VI variants to add and explain the C++/Python VIGS performance gap.
- [ ] Decide whether VISOR needs additional registered configurations beyond its constructor options.
- [ ] Select other solvers to add.
- [ ] Define a method for tuning solver parameters.
- [ ] Compare MDP formulations as constrained linear programs against standard MDP methods, identifying the primal or dual formulation and optimization backend.
- [ ] Explain solver performance through model structure: action counts, transition nonzeros, cycles and terminal reachability at comparable sizes. Define the experimental route later.
- [ ] Compare aggregated methods with their non-aggregated baselines using runtime, solution quality and memory use to identify when aggregation pays off and whether it saves representation space. Distinguish input storage, aggregate representation and temporary storage from peak process memory; define memory profiling separately from runtime measurements.
- [ ] Plan experiments against standard methods, identify feasible improvements and demonstrate a scientific contribution. See the [visualization roadmap](visualization-roadmap.md) for measurements and README additions.

## MPI inner-precision study

- [ ] Start with discounted MPI implementations exposing an inner tolerance; record each option's meaning and scaling. The current total-reward `mdpforge_mpi` exposes an inner iteration budget but no `evaluation_precision`, so study its budget separately.
- [ ] Sweep `evaluation_precision` on a logarithmic grid, including the default, at fixed final precision and inner iteration budget; keep model instances, paired seeds and other solver settings fixed.
- [ ] Repeat these sweeps for selected discounts, final precision targets and inner iteration budgets, changing one setting between sweeps to identify interactions rather than mixing their effects.
- [ ] Compare runtime, independently checked final quality, failures and timeouts. Distinguish inner threshold attainment from iteration-budget exhaustion where observable; report unknown stopping reasons as unknown. Plan any missing diagnostics separately while preserving existing solver internals and timing boundaries.
- [ ] Report when extra inner work reduces total runtime, when tighter thresholds have no effect because the budget binds, and whether a useful default depends on discount, final precision or model family. Treat identical outputs or runtimes alone as insufficient evidence that the budget binds.
