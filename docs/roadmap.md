# Open questions

- [ ] Test that each solver returns a valid and accurate `SolveResult`.
- [ ] Add an output-precision check for the value function in the benchmark, with an automated test that validates this check.
- [ ] Stop exposing `value`, `policy`, `runtime`, `iterations` and `stop_reason` as solver result attributes; migrate solvers with custom `run()` methods, the benchmark, the cache and other callers to `SolveResult`.
- [ ] Simplify the benchmark class by moving functions into utilities or splitting responsibilities across classes.
- [ ] List and add VI variants; clarify the VIGS performance gap between C++ and Python.
- [ ] Support several VISOR configurations?
- [ ] Define the remaining minimum test coverage: each model's arguments and useful cases beyond the shared contract.
- [ ] Get madupite working; none of its solvers currently work.
- [ ] Shorten the Markdown documentation where possible.
- [ ] Which other solvers should be added?
- [ ] How should MDP solver parameters be tuned?
- [ ] Translate the documentation into English and shorten it to the strict minimum. Translate the roadmap.
- [ ] Plan experiments: list standard solvers/approaches, identify improvements that can actually be implemented, and add a contribution.
- [ ] Should tests check the final value function or the Bellman residual?
