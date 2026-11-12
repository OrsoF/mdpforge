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
- [ ] Should tests check the final value function or the Bellman residual?
- [ ] Quels paramètres devraient aller dans Metadata, lesquels directement dans le solveur ou le modèle ?
- [ ] Comment rendre universel le format des solveurs, avec tous leurs paramètres ? un dictionnaire d'options ? avec seulement une precision finale ?


- [ ] Observe le code de rooms.py, c'est du code perso. Observe le style, je veux que ton code suive un style similaire. Quels sont les principes appliqués dans ce style ? Ajoute ces principes à AGENTS
- [ ] Le code de rooms.py contient n'est pas non plus le meilleur: il contient des redondances, des asserts etc. Comment le rendre plus compact en gardant son style compréhensible ? Quels réorganisation tu envisagerais pour le rendre encore plus compact et plus clair ?
- [ ] Souvent, dans le style de tes réponses, je ne comprends pas ce que tu veux dire. Garde des termes simples, des explications claires. Ajoute à AGENTS
- [ ] Ok, tu as pris des initiatives pour la transposition des modèles au cas total. Pour ninerooms.py, ce qu'on va faire: 1) reproduire EXACTEMENT le modèle discounted dans le dossier total (oublie épisodic etc). Met la reward à 0 partout. Met une reward de 1 dans le cas unique: si (s, a) conduit avec probabilité > 0 à l'état absorbant. 
- [ ] On considère des MDPs cas total. Dans tous mes MDPs, je voudrais inclure la reward suivante:
R(s, a) = {0 si T(s, a, s_final) = 0, 1 otherwise}. Est-ce que cette reward vérifie le critère de Bertsekas et peut assurer la convergence de VI/PI/MPI quoiqu'il arrive ? Crée un plan pour vérifier cela rigoureusement mathématiquement
- [ ]
