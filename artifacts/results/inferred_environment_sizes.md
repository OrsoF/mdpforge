# Tailles large inférées

Source : [model_sizes.csv](model_sizes.csv). Les anciennes lignes large sont exclues.

Cible : 30 s, prévision plafonnée à 15 s pour garder une marge. Même γ = 0,99, précision 10⁻³ et un thread. Les paramètres small/medium restent ceux mesurés ; ils ne sont pas recalibrés ici.

Pour chaque solveur ayant réussi aux deux tailles, on ajuste t = c × Wᵝ. W = A × S² pour mdptoolbox ; W varie comme le nombre de transitions pour les autres. β est au moins 1. Parmi les candidats examinés, on retient le plus grand palier constructeur dont toutes les prévisions ajustables restent sous 15 s. Les cibles d'états sont espacées de 20 % de la taille medium avant adaptation aux paliers natifs (pistes entières pour Sutton). Limites : huit fois les états medium, 25 000 états, 128 actions, matrices estimées sous 256 Mio, construction estimée sous 60 s.

Une seule répétition par point : estimations heuristiques, sans intervalle de confiance. Les algorithmes autres que VI et les backends en échec ne sont pas couverts. Aucun solveur ni matrice large n'a été exécuté/construit.

Configurations utilisables : [inferred_environment_sizes.json](inferred_environment_sizes.json), champ `libraries`. Valeur `null` : configuration indisponible ou inférence impossible. Les modèles et caches existants sont inchangés.

| Modèle | États small → medium | Demande constructeur large | États / actions large | Prévision max (s) | Note |
|---|---:|---:|---:|---:|---|
| access_control | 100 → 500 | 1690000 | 1300 / 2 | 13.57 | Extrapolation prudente, non mesurée. Budget valable seulement pour les solveurs ajustables ayant réussi aux deux tailles. Échecs CSV : mdptoolbox_vi. |
| acrobot | 257 → 1297 | 5231 | 6562 / 3 | 12.21 | Extrapolation prudente, non mesurée. |
| admission | 108 → 1008 | 7872 | 7866 / 2 | 12.88 | Extrapolation prudente, non mesurée. |
| ambulance | 100 → 200 | 160000 | 400 / 10 | 13.11 | Extrapolation prudente, non mesurée. Budget valable seulement pour les solveurs ajustables ayant réussi aux deux tailles. Échecs CSV : mdptoolbox_vi. |
| ambulance_relocation | 50 → 200 | 129600 | 360 / 10 | 12.76 | Extrapolation prudente, non mesurée. Budget valable seulement pour les solveurs ajustables ayant réussi aux deux tailles. Échecs CSV : mdptoolbox_vi. |
| barto | 146 → 726 | 4385 | 3856 / 9 | 11.43 | Extrapolation prudente, non mesurée. |
| block | 100 → 400 | 1200 | 1200 / 10 | 13.41 | Extrapolation prudente, non mesurée. |
| blocks_world | 96 → — | — | — | — | Dimension fixe : un seul point exploitable. Échecs CSV : unavailable_size. |
| car_rental | 25 → 64 | 147456 | 400 / 9 | 14.81 | Extrapolation prudente, non mesurée. |
| cartpole | 256 → 625 | 4555 | 4096 / 2 | 0.33 | Extrapolation prudente, non mesurée. Budget valable seulement pour les solveurs ajustables ayant réussi aux deux tailles. Échecs CSV : mdptoolbox_vi. |
| chain_walk | 100 → 1000 | 8000 | 8000 / 2 | 12.92 | Extrapolation prudente, non mesurée. |
| cliff | 75 → 972 | 6482 | 6348 / 4 | 10.74 | Extrapolation prudente, non mesurée. |
| dam | 16 → 81 | 454 | 625 / 25 | 5.68 | Extrapolation prudente, non mesurée. |
| elevator | 272 → 2176 | — | — | — | Le constructeur ne propose que 3 ou 4 étages ; medium est déjà le maximum. |
| forest | 100 → 1000 | 8000 | 8000 / 2 | 8.71 | Extrapolation prudente, non mesurée. |
| frozen | 100 → 961 | 6192 | 6084 / 4 | 14.46 | Extrapolation prudente, non mesurée. |
| gambler | 101 → 1001 | 4627 | 4628 / 10 | 14.34 | Extrapolation prudente, non mesurée. |
| garnet | 100 → 200 | 400 | 400 / 10 | 13.85 | Extrapolation prudente, non mesurée. |
| hexagonal_grid_soccer | 343 → 6859 | — | — | — | Aucun palier supérieur ne respecte la marge et les limites d'extrapolation. |
| impatience | 100 → 300 | 1800 | 1800 / 10 | 14.66 | Extrapolation prudente, non mesurée. |
| inventory | — → — | — | — | — | Le script lit state_dim avant create_model(), alors que cette recette le définit pendant la construction. Aucune mesure exploitable. Échecs CSV : build_error. |
| inventory_leadtime | 100 → 1000 | 211600 | 4600 / 10 | 14.48 | Extrapolation prudente, non mesurée. |
| local_perception_grid_world | 100 → 1023 | 6674 | 6723 / 4 | 14.73 | Extrapolation prudente, non mesurée. |
| lqr | 100 → 1000 | 4600 | 4600 / 10 | 14.31 | Extrapolation prudente, non mesurée. |
| m_maze | 100 → 1023 | 6674 | 6723 / 4 | 14.97 | Extrapolation prudente, non mesurée. |
| maintenance | 100 → 1000 | 7000 | 7000 / 4 | 14.19 | Extrapolation prudente, non mesurée. |
| maze_backtrack | 100 → 961 | 7033 | 6889 / 4 | 14.58 | Extrapolation prudente, non mesurée. |
| maze_prims | 100 → 961 | 7454 | 7396 / 4 | 14.63 | Extrapolation prudente, non mesurée. |
| maze_wilson | 100 → 961 | 7454 | 7396 / 4 | 14.73 | Extrapolation prudente, non mesurée. |
| mountain | 100 → 961 | 7665 | 7569 / 3 | 11.51 | Extrapolation prudente, non mesurée. |
| ninerooms | 81 → 900 | 7130 | 7056 / 4 | 13.73 | Extrapolation prudente, non mesurée. |
| office_world | 100 → 1020 | 6885 | 6888 / 4 | 14.65 | Extrapolation prudente, non mesurée. |
| oil | 100 → 1000 | 4800 | 4800 / 10 | 14.51 | Extrapolation prudente, non mesurée. |
| oil_discovery | 100 → 300 | 840 | 840 / 10 | 13.35 | Extrapolation prudente, non mesurée. |
| option_pricing | 101 → 962 | 7727 | 7570 / 2 | 8.06 | Extrapolation prudente, non mesurée. |
| peg_solitaire | — → 511 | — | — | — | Un seul point exploitable ; le prochain plateau passe de 511 à 65 535 états. Échecs CSV : unavailable_size. |
| queue_network | 125 → 1000 | 5943 | 6859 / 3 | 11.71 | Extrapolation prudente, non mesurée. |
| random_walk | 100 → 1000 | 8000 | 8000 / 2 | 10.65 | Extrapolation prudente, non mesurée. |
| replacement | 100 → 1000 | 4200 | 4200 / 10 | 13.78 | Extrapolation prudente, non mesurée. |
| rooms | 100 → 900 | 6767 | 6724 / 4 | 13.62 | Extrapolation prudente, non mesurée. |
| schoolboy | 100 → 1000 | 8000 | 8000 / 3 | 12.74 | Extrapolation prudente, non mesurée. |
| sutton | 283 → 1231 | 1000 | 4211 / 9 | 10.79 | Extrapolation prudente, non mesurée. Piste carrée 11 × 11. |
| swim | 100 → 1000 | 8000 | 8000 / 2 | 11.10 | Extrapolation prudente, non mesurée. |
| sysadmin | 64 → 512 | 2048 | 2048 / 11 | 5.46 | Extrapolation prudente, non mesurée. |
| tandem | 100 → 900 | 4500 | 4624 / 9 | 14.66 | Extrapolation prudente, non mesurée. |
| tandem_choice | 36 → 100 | 620 | 625 / 25 | 4.78 | Extrapolation prudente, non mesurée. |
| taxi | 81 → 981 | 5869 | 5781 / 6 | 13.80 | Extrapolation prudente, non mesurée. |
| unichain | 100 → 1000 | 8000 | 8000 / 2 | 10.98 | Extrapolation prudente, non mesurée. |
| windgrid | 100 → 1023 | 7451 | 7482 / 4 | 14.48 | Extrapolation prudente, non mesurée. |
| wumpus_world | 1024 → 2304 | 2915 | 6400 / 6 | 10.32 | Extrapolation prudente, non mesurée. |
