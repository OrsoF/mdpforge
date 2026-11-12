# Open questions

- [ ] Test that each solver returns a valid and accurate `SolveResult`.
- [ ] Add an output-precision check for the value function in the benchmark, with an automated test that validates this check.
- [ ] Stop exposing `value`, `policy` and `runtime` as solver result attributes; migrate solvers with custom `run()` methods, the benchmark, the cache and other callers to `SolveResult`.
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

- [ ] Ok, tu as pris des initiatives pour la transposition des modèles au cas total. Pour ninerooms.py, ce qu'on va faire: 1) reproduire EXACTEMENT le modèle discounted dans le dossier total (oublie épisodic etc). Met la reward à 0 partout. Met une reward de 1 dans le cas unique: si (s, a) conduit avec probabilité > 0 à l'état absorbant. 
- [ ] warning: in the working copy of 'src/mdpforge/models/total/rooms.py', LF will be replaced by CRLF the next time Git touches it. ça veut dire quoi ce warning ? COmment faire en sorte qu'il n'apparaissent plus ? (sans supprimer directement le warning, je dois changer une manière de coder ?)
- [ ] Ajouter tous les autres solveurs, dont aggvi, aggqvi, aggmpi. Ajouter un test de tous les solveurs total sur une instance rooms 100.
- [ ] Ajouter le modèle sysadmin en total, vérifier quels modeles autres on peut ajouter 