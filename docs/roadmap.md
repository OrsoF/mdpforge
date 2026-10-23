# Open questions

- [ ] Tester que chaque solveur retourne un `SolveResult` valide et précis.
- [ ] Ne plus exposer `value`, `policy`, `runtime`, `iterations` et `stop_reason` comme attributs de résultat des solveurs ; migrer les solveurs à `run()` propre, le benchmark, le cache et les autres appelants vers `SolveResult`.
- [ ] Alléger la classe bench, ou avec des fonctions dans utils, ou bien en séparant en plusieurs instances
- [ ] Pourquoi ça ne tourne pas en local ? OSError c'esst quoi ce truc
- [ ] Lister et ajouter des variantes de VI ; éclaircir l'écart de performance VIGS entre C++ et Python.
- [ ] Plusieurs paramétrages pour visor ?
- [ ] Pourquoi git se lance pas dans le terminal local ?
- [ ] Définir la couverture minimale restante des tests : arguments de chaque modèle et cas utiles au-delà du contrat commun.
- [ ] Faire fonctionner madupite, pour l'instant aucun solveur ne fonctionne
- [ ] Raccourcir les docs md si possible.
- [ ] D'autres solveurs à ajouter ?
- [ ] Comment on finetune un paramètres dans un solveur MDP ?
- [ ] Traduire en anglais et raccourcir la doc à son strict minimum. Traduire et détailler la roadmap
- [ ] Expériences à lancer: faire une liste des solveurs/approches usuelles pour savoir quelles améliorations peuvent être réellement implémentées et ajouter quelque chose.
