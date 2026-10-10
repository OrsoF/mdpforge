# Vérification des tailles du catalogue

Vérification du 10 octobre 2026, avec les paramètres de `METADATA` des fichiers
modèles. Les rapports `inferred_environment_sizes.*` décrivent la première
inférence et ne constituent plus la configuration actuelle du catalogue.

## Résultats

- 50 modèles ; 50 configurations small, 50 medium, 49 large disponibles.
- 149 configurations : dimension du constructeur conforme à la dimension déclarée.
- Une configuration indisponible : `hexagonal_grid_soccer/large`.
- Les 12 configurations de Blocks World, Elevator, Peg Solitaire et Inventory
  ont été construites sans cache de référence et validées : dimensions, CSR,
  récompenses finies et transitions stochastiques.
- Vérification du parcours de résolution sur Unichain small avec mdpforge VI :
  succès et borne d'erreur indépendante respectant `1e-3` à discount `0.99`.

Les vérifications des constructeurs et des matrices ne mesurent pas les temps
des solveurs. Les temps des tailles large et des modèles dont la dynamique a
changé restent à mesurer. En particulier, Blocks World large compte 96 000 états :
sa dimension est valide, mais le budget de 30 secondes n'est pas établi.

| Modèle modifié ou actualisé | Small | Medium | Large | Origine des dimensions |
|---|---:|---:|---:|---|
| blocks_world | 96 | 9 600 | 96 000 | Mesure historique small ; extension paramétrée medium/large |
| elevator | 176 | 1 664 | 14 592 | Calcul analytique ; temps actuels non mesurés |
| peg_solitaire | 128 | 512 | 8 192 | Configurations explicites ; temps actuels non mesurés |
| inventory | 99 | 999 | 4 595 | Small/medium mesurés ; large inférée |

Les anciennes raisons d'indisponibilité d'Inventory ont été remplacées par les
mesures réussies du CSV. Large vise une prévision maximale de 14,50 s, avec une
marge pour le budget de 30 s. Voir [la provenance](inventory_size_inference.json).

Hexagonal Grid Soccer conserve un palier suivant de 50 653 états après medium
(6 859 états), exclu de l'inférence initiale. Son label large est encore absent
et aucun problème de constructeur n'a été détecté pour small/medium.

## Reproduire et compléter les vérifications

```text
conda run -n benchmark python check_model_sizes.py --dimensions-only
conda run -n benchmark python check_model_sizes.py --build-only --sizes small medium large --models blocks_world elevator peg_solitaire inventory
conda run -n benchmark python check_model_sizes.py --sizes large
```

Le dernier lancement est destiné à mesurer les temps réels des solveurs sur les
tailles large, pas à être exécuté automatiquement pendant cet audit. Ajouter
`--sizes small medium large` pour mesurer toutes les tailles, ou `--models` pour
limiter le lancement. Les budgets par taille sont 1, 5 et 30 s ; chaque solveur
a un timeout de 60 s par défaut et une vérification indépendante du résidu.

Les résultats sont enregistrés progressivement et les cas terminés sont repris
au lancement suivant. Le code de sortie 1 signale un problème identifié, y compris
une taille indisponible ou un dépassement du budget. Une modification du fichier
modèle crée de nouveaux cas grâce à son empreinte source ; les lignes historiques
restent dans le CSV.

Résultats : [dimensions](model_size_dimensions.csv) et
[constructions](model_size_build.csv). La syntaxe des nouveaux scripts a été
vérifiée. Pytest et Ruff n'ont pas été exécutés ; cet audit ne remplace pas la suite
complète ou la CI.

## Nettoyage

135 fichiers temporaires supprimés sous `artifacts/tmp/` : anciens scripts de
génération/calibration remplacés et modèles sérialisés des calibrations arrêtées.
Environ 219 Mio libérés. Les CSV, rapports JSON, journaux de diagnostic et autres
résultats de référence sont conservés.
