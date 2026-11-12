# Roadmap

## Interfaces and benchmarks

- [ ] Finish migrating solvers and callers to `SolveResult`; remove solver result attributes `value`, `policy` and `runtime`.
- [ ] Simplify `Benchmark` by moving helpers into utilities or separating responsibilities.
- [ ] Decide which parameters belong in `METADATA` or model/solver constructors, and whether solvers need a shared options format and final precision target.

## Tests and backends

- [ ] Test the benchmark's optional output-precision validation.
- [ ] Define model-specific test coverage beyond the shared contract, including arguments and useful edge cases.
- [ ] Diagnose and resolve madupite solver failures.

## Algorithms and experiments

- [ ] Review VI variants to add and explain the C++/Python VIGS performance gap.
- [ ] Decide whether VISOR needs additional registered configurations beyond its constructor options.
- [ ] Select other solvers to add.
- [ ] Define a method for tuning solver parameters.
- [ ] Plan experiments against standard methods, identify feasible improvements and demonstrate a scientific contribution. See the [visualization roadmap](visualization-roadmap.md) for measurements and README additions.
