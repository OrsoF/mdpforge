# README and visualization roadmap

The README should help a visitor answer two questions quickly: "Can I test my
MDP here?" and "Can I compare my solver here?" Its opening now offers both paths,
a compact comparison with repeat spread, and a link to an executed visual gallery.

## Views available in the exploratory notebook

[`notebooks/visualize_results.ipynb`](../notebooks/visualize_results.ipynb) reads
saved benchmark exports without executing solvers. Its functions are intentionally
local to this notebook while the designs are explored; the library's existing
`BenchmarkResult.plot_heat()` offers the library's runtime/speedup heatmap.

| View | Best use | Interpretation |
| --- | --- | --- |
| Paired speedups | First README figure; compare a new solver on a few MDPs | Higher is faster; each trial uses the same repeat and seed as its reference |
| Performance profiles | Compare many solvers on many MDPs | Fraction of all models solved within a given factor of the fastest eligible median |
| Trial outcomes and full-model coverage | Show robustness next to speed | One failed repeat excludes that solver/model pair from the runtime comparison |
| Runtime vs error bound | Explain shared accuracy and diagnose oversolving | A tighter certified bound is not a measurement of actual value or policy error |

Use the small README export to illustrate model-dependent rankings. The larger
backend export was recorded on 2026-10-09 with MDPSolver 0.10.1; its old VISOR
errors are historical. Regenerate that export before making claims about current
backend performance. Keep the CSV and matching experiment manifest with every
published comparison. A solver-set change changes a performance profile's
reference minima; do not compare profiles built from different candidate sets
as though they used the same baseline.

## Suggested next additions to the README

1. **Two short contribution walkthroughs.** Show one external model entering
   the benchmark and one external solver producing a comparison. The current
   quickstart already demonstrates both registrations. Add a dedicated solver
   contribution guide before implying that such a guide exists.
2. **An MDP catalogue preview.** A small table of navigation, physical-control,
   service-operations and abstract examples, with actual size presets and recipe links,
   would make the collection easier to discover. Read catalogue metadata rather
   than maintaining a manually counted headline.
3. **A current broad comparison.** Once refreshed, pair the performance profile
   with the coverage figure. Put full solver legends and settings in the notebook;
   keep only a readable subset in the README, documenting the selection.
4. **One worked scientific contribution.** Explain an MDP structure, why one
   solver exploits it, and which accuracy checks support the result. This gives
   potential contributors a concrete standard beyond a runtime leaderboard.

## Next plots that need new measurements

The [solver sensitivity notebook](../notebooks/solver_sensitivity.ipynb) now
provides a measured small-instance example for method comparisons, discount
sweeps and precision sweeps. It reuses each model's matrices, records per-setting
experiment manifests, and supports a separate full-grid experiment. The saved
demo varies one factor at a time and does not estimate interactions or scaling.

| Plot | Experiment needed | Why it helps |
| --- | --- | --- |
| Runtime vs states or transition nonzeros | Several sizes of the **same** model family, fixed solver settings, precision, discount and thread counts | Reveals scaling without confusing model structure with size |
| Runtime vs discount | Reuse the exact matrices; sweep discount with a shared absolute error target | Shows which solvers handle long horizons well |
| Runtime vs requested precision | Reuse matrices and discount; sweep target precision | Compares the cost of tighter accuracy |
| Bellman-residual convergence | Record iteration or elapsed-time traces, with documented measurement overhead | Explains how algorithms make progress, rather than just their final time |

The final-result CSVs do not contain convergence traces. The size-calibration
export contains different model families at largely one selected size each, so
it cannot justify a single scalability curve. Avoid drawing either curve from
those records. The visual gallery reads historical measurements; the separate
sensitivity notebook has its own freshly collected small-instance study.
