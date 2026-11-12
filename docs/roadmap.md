# Open questions

- [ ] Add an automated test for the benchmark's existing output-precision check.
- [ ] Finish migrating solvers with custom `run()` methods and remaining callers to `SolveResult`; stop exposing `value`, `policy` and `runtime` as solver result attributes.
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


- [ ] Quels paramètres devraient aller dans Metadata, lesquels directement dans le solveur ou le modèle ?
- [ ] Comment rendre universel le format des solveurs, avec tous leurs paramètres ? un dictionnaire d'options ? avec seulement une precision finale ?

- [ ] Normalize local Python files to LF to match the existing `.gitattributes` rule and resolve line-ending warnings.
