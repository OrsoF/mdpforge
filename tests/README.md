# Tests de mdpforge

La suite sépare les **contrats communs**, les **propriétés scientifiques** et les **régressions spécifiques**. Les tests ne doivent pas dupliquer l'algorithme qu'ils vérifient.

## Ajouter un solveur

Le catalogue est découvert par `tests/conftest.py`. Un module `src/mdpforge/solvers/<nom>.py` est automatiquement soumis aux tests de `tests/test_workflows.py` : précision de la fonction de valeur via un oracle indépendant, résidu de Bellman, politique optimale, reproductibilité et conservation des données d'entrée. L'oracle énumère les politiques sur des MDP suffisamment petits pour rester indépendant des solveurs de production ; ses identités mathématiques sont vérifiées dans `tests/test_scientific_properties.py`.

Les backends optionnels sont exécutés en environnement isolé uniquement si leurs dépendances sont présentes. Les variantes de madupite utilisent une seule simulation de backend dans `tests/_fake_backends.py` ; les propriétés PETSc de chaque profil sont déclarées dans une table. Les deux solveurs MDPax partagent le même contrôle de sorties.

Un test spécifique reste justifié pour une propriété **non couverte** par l'interface commune : cas pathologique, conversion externe, tolérance numérique interne, partitionnement, cache, timeout ou export. Conserver un contre-exemple concret lorsqu'il empêche le retour d'une régression réelle.

## Commandes

Après installation du projet en mode développement et des dépendances de test : `python -m pytest`. Les backends natifs ne sont testés que s'ils sont installés ; les tests simulés madupite/MDPax fonctionnent sans ces dépendances. Pour parcourir la liste de tests sans les exécuter : `python -m pytest --collect-only -q`. Les fichiers `tests/_contracts.py` et `tests/_fake_backends.py` sont des utilitaires de test, pas des tests à lancer directement.
