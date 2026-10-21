# Open questions

- [ ] Unformiser l'output des solveurs: trouver 1) ce qu'ils doivent output, 2) un test qui le vérifie
- [ ] Ne plus exposer `value`, `policy`, `runtime`, `iterations` et `stop_reason` comme attributs de résultat des solveurs : `_solve()` retourne un `SolveResult` avec le résultat du calcul, `run()` y ajoute le temps d'exécution et retourne le résultat complet ; adapter les backends, le benchmark, le cache et les autres appelants sans changer les algorithmes.
- [ ] Nouvelle catégorie vi_exp pour les solveurs aditionnels vi ? Non, autre catégorisation, un peu plus précise
- [ ] Alléger la classe bench, ou avec des fonctions dans utils, ou bien en séparant en plusieurs instances
- [ ] Pourquoi ça ne tourne pas en local ? OSError c'esst quoi ce truc
- [ ] Lister et ajouter des version alternatives de VI. Pourquoi VIGS tourne bien sous C++ et pas sous python ?
- [ ] Plusieurs paramétrages pour visor ?
- [ ] Auditer les tests: est-ce qu'ils vont trop loin ?
- [ ] Pourquoi git se lance pas dans le terminal local ?
