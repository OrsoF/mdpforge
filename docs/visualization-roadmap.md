# Visualization roadmap

The [README](../README.md) introduces model and solver registration. The [visual gallery](../notebooks/visualize_results.ipynb) explains paired speedups, performance profiles, trial outcomes and runtime/error-bound plots using saved CSVs.

## README additions

- [ ] Add two short walkthroughs: register an external model and compare an external solver. Link the existing [model](adding-a-model.md) and [solver](adding-a-solver.md) guides.
- [ ] Preview the model catalogue with representative tasks, available sizes and recipe links drawn from metadata.
- [ ] Refresh the broad comparison and pair its performance profile with full-repeat coverage. Keep complete settings in the experiment manifest and explain any legend selection.
- [ ] Explain one scientific contribution: model structure, the solver that exploits it and the checks supporting the result.

## Measurements needed

| Plot | Controlled experiment |
| --- | --- |
| Runtime vs states or transition nonzeros | Several sizes of one model family; fixed solver settings, precision, discount and thread counts. |
| Runtime vs discount | Identical matrices; shared absolute error target. |
| Runtime vs requested precision | Identical matrices and discount; vary the error target. |
| Bellman-residual convergence | Iteration or elapsed-time traces; document measurement overhead. |

Keep a CSV and matching experiment manifest for each setting, including validation mode. The saved sensitivity demo varies one factor at a time; it does not measure interactions or scaling. Final-result CSVs have no convergence traces, and the size-calibration export mixes families at largely one selected size each.

## Interpreting saved results

The small README sample illustrates model-dependent rankings. The broad backend export dates from 2026-10-09 with MDPSolver 0.10.1; its VISOR failures are historical. Regenerate it before drawing conclusions about current performance.

Runtime comparisons require every expected repeat to succeed. Performance profiles retain all models in their denominator and use the fastest eligible median across the selected solvers; changing that set changes the baseline. A certified error bound is not the measured value or policy error. Current benchmarks certify results only with `validate=True`; keep validation status separate from execution success.
