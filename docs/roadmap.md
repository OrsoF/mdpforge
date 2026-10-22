# Open questions

- [ ] Unformiser l'output des solveurs: trouver 1) ce qu'ils doivent output, 2) un test qui le vérifie
- [ ] Ne plus exposer `value`, `policy`, `runtime`, `iterations` et `stop_reason` comme attributs de résultat des solveurs : `_solve()` retourne un `SolveResult` avec le résultat du calcul, `run()` y ajoute le temps d'exécution et retourne le résultat complet ; adapter les backends, le benchmark, le cache et les autres appelants sans changer les algorithmes.
- [ ] Nouvelle catégorie experimentale pour les solveurs
- [ ] Alléger la classe bench, ou avec des fonctions dans utils, ou bien en séparant en plusieurs instances
- [ ] Pourquoi ça ne tourne pas en local ? OSError c'esst quoi ce truc
- [ ] Lister et ajouter des version alternatives de VI. Pourquoi VIGS tourne bien sous C++ et pas sous python ?
- [ ] Plusieurs paramétrages pour visor ?
- [ ] Auditer les tests: est-ce qu'ils vont trop loin ?
- [ ] Pourquoi git se lance pas dans le terminal local ?
- [ ] Supprimer tous les tests: écrire seulement ceux les plus importants. En particulier: 
    - test de l'out de chaque solveur: a precision crtierion. Est-ce que le mettrait pas dans SolveResult directement ?
    - Test les arguments de chaque modèle: shape, type, etc. 
    - Refonder entièrement la procédure de test: en dehors de ces tests énoncés, demander lesquels peuvent être utiles. 
- [ ] Faire fonctionner madupite, pour l'instant aucun solveur ne fonctionne
- [ ] Raccourcir les docs md si possible.
- [ ] D'autres solveurs à ajouter ?
- [ ] Comment on finetune un paramètres dans un solveur MDP ?

