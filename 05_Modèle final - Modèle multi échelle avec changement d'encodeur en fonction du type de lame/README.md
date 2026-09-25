# 05. Modèle final : multi-échelle avec changement d'encodeur selon le type de lame

Ce dossier contient le meilleur modèle du stage (« double encodeur ») et son évaluation sur des
données externes (CAMELYON).

| Script | Rôle |
|---|---|
| [`MIL_separe_double_encodeur.py`](MIL_separe_double_encodeur.py) | entraîne 2 modèles et compare 3 approches sur le même test (modèle final) |
| [`MIL_separe_double_encodeur_CAMELYON.py`](MIL_separe_double_encodeur_CAMELYON.py) | même chose, avec les lames CAMELYON ajoutées au jeu de base |
| [`Modele_final_TEST.py`](Modele_final_TEST.py) | inférence seule d'un modèle déjà entraîné sur CAMELYON (test externe) |

Le nom du dossier contient des espaces et une apostrophe : lancer les scripts entre guillemets
depuis la racine du dépôt :

```bash
conda activate atlas_patch
D="05_Modèle final - Modèle multi échelle avec changement d'encodeur en fonction du type de lame"
python "$D/MIL_separe_double_encodeur.py"
python "$D/MIL_separe_double_encodeur_CAMELYON.py"
python "$D/Modele_final_TEST.py"
```

---

## Idée du double encodeur

Les résultats précédents montrent que :
- sur les biopsies, le meilleur modèle est le MIL 20x seul avec H-optimus-1 ;
- sur les pièces chirurgicales, le 5x aide, et Virchow2 (pré-entraîné à plusieurs
  grossissements) est le meilleur encodeur en multi-échelle.

On garde donc le meilleur des deux, en routant chaque lame selon son type :

```
                 ┌── type == "Biopsy" ──► MIL 20x          (features H-optimus-1, 1536)
lame de test ────┤
                 └── pièce chirurgicale ─► MIL concat 20x+5x (features Virchow2, 2560)
```

Les deux modèles sont entraînés séparément, chacun sur toutes les lames du train. Le routage
n'intervient qu'à la prédiction.

---

## `MIL_separe_double_encodeur.py`

### Données nécessaires

Pour chaque lame, trois jeux de features, avec le même split :

| Manifest | Encodeur | Échelle |
|---|---|---|
| `HOPT_MANIFEST_20X` | `h_optimus_1` (1536) | 20x |
| `VIR_MANIFEST_20X` | `virchow_v2` (2560) | 20x |
| `VIR_MANIFEST_5X` | `virchow_v2` (2560) | 5x |

Soit trois encodages AtlasPatch (voir le [README principal](../README.md#54-enchaîner-plusieurs-encodages-20x-et-5x)),
puis `Label_and_split.py` sur les deux dossiers 20x et `Manifest_5x.py` pour le 5x.

### Déroulement

1. Ne garde que les lames présentes dans les trois manifests.
2. Vérifie que le split est identique entre H-optimus et Virchow2 ; sinon le script s'arrête
   (`[!] N lames avec split different -> routage invalide`), car une lame pourrait être en train
   pour un modèle et en test pour l'autre.
3. Entraîne le MIL 20x H-optimus-1 (`MIL20x`), puis le MIL concaténé Virchow2
   (`MILConcat`). Pour chacun : `ReduceLROnPlateau`, meilleure epoch (accuracy val puis AUC),
   early stopping, courbes, sauvegarde du modèle.
4. Évalue trois approches sur le même test :
   1. H-optimus-1 20x seul (toutes les lames) ;
   2. Virchow2 concaténé 20x+5x (toutes les lames) ;
   3. double encodeur (routage selon `type`).
5. Pour chacune : lames mal classées (avec type et diagnostic), accuracy, AUC, matrice de confusion,
   rapport de classification ; puis un résumé comparatif.

### Sorties (dans `DOSSIER_SORTIE`)

| Fichier | Contenu |
|---|---|
| `courbes_hopt20x.png`, `courbes_vir_concat.png` | courbes d'apprentissage des deux modèles |
| `modele_hopt20x.pth`, `modele_vir_concat.pth` | poids des deux modèles (meilleure epoch) |
| `matrice_hopt20x.png`, `matrice_vir_concat.png`, `matrice_double_encodeur.png` | matrices de confusion des 3 approches |

### À modifier

| Variable | Rôle |
|---|---|
| `HOPT_MANIFEST_20X`, `VIR_MANIFEST_20X`, `VIR_MANIFEST_5X` | les 3 manifests |
| `HOPT_ENCODEUR` / `HOPT_DIM`, `VIR_ENCODEUR` / `VIR_DIM` | encodeurs utilisés (on peut en essayer d'autres, ex. UNI, CONCH) |
| `DOSSIER_SORTIE` | figures + modèles |
| `TYPE_BIOPSIE` | libellé exact des biopsies dans la colonne `type` (`"Biopsy"`) |
| `NB_EPOCHS` (100), `PATIENCE` (30), `TAUX_APPRENTISSAGE`, `MAX_PATCHS_PAR_LAME` (2000), `SEED` (78) | entraînement |

Pour inverser le routage ou en ajouter un autre (ex. selon la `cohort`), modifier la boucle
de l'approche 3 (`if r["type"] == TYPE_BIOPSIE:` dans `main()`).

### Résultats (jeu lymphome folliculaire, test = 130 lames)

| Approche | Accuracy | AUC |
|---|---|---|
| H-optimus-1 20x seul | 0,946 | 0,988 |
| Virchow2 concaténation 20x + 5x | 0,931 | 0,993 |
| Double encodeur | 0,962 | 0,989 |

Le double encodeur fait 5 erreurs sur 130 lames, dont 2 lymphomes manqués sur 75.

---

## `MIL_separe_double_encodeur_CAMELYON.py`

Même script que le précédent, mais les lames CAMELYON (toutes étiquetées FH) sont ajoutées
au jeu de base (FH + FL) avant l'entraînement : `charger_et_fusionner` concatène, pour chaque
encodeur, le manifest de base et le manifest CAMELYON, et ajoute une colonne `source`
(`LF` / `CAMELYON`).

Différences avec la version de base :
- s'arrête si deux lames des deux sources ont le même nom (`Collision de noms de lames`) ;
- affiche la répartition label × split, pour vérifier que val et test contiennent bien FH ET FL ;
- `classification_report` et `roc_auc_score` protégés contre le cas d'une seule classe.

Données CAMELYON à préparer : encodages H-optimus-1 20x, Virchow2 20x et Virchow2 5x (commande
`nohup bash -c '…'` du README principal), puis `Label_and_split.py` et `Manifest_5x.py` sur ces
dossiers, avec un CSV de labels CAMELYON au même format (colonne `type` comprise).

À modifier : les chemins `BASE_*` (jeu de base), `HOPT_MANIFEST_20X`, `VIR_MANIFEST_20X`,
`VIR_MANIFEST_5X` (CAMELYON), `DOSSIER_SORTIE`, `NB_EPOCHS` (200), `PATIENCE` (45), `SEED`.

> ⚠ Même noms de fichiers de sortie que la version de base (`modele_hopt20x.pth`, …) : utiliser
> un `DOSSIER_SORTIE` différent pour ne pas écraser le modèle final.
>
> ⚠ Les manifests CAMELYON H-optimus et Virchow2 doivent avoir le même split : lancer
> `Label_and_split.py` avec le même CSV et la même graine, et vérifier que le même nombre de lames
> est apparié pour les deux encodeurs.

---

## `Modele_final_TEST.py` : test externe sur CAMELYON

Aucun entraînement : le script charge un modèle `.pth` déjà entraîné et prédit chaque lame du
manifest CAMELYON (tous les patchs sont utilisés, `MAX_PATCHS_PAR_LAME = None`, donc le résultat
est entièrement reproductible).

Sorties : lames mal classées, accuracy, matrice `matrice_camelyon_MIL.png` et un CSV
`predictions_camelyon.csv` (une ligne par lame : `slide, vrai, type, predit, score_FL`).

### Points importants

- Anti-fuite : le modèle chargé ne doit jamais avoir vu CAMELYON. Ne pas utiliser un
  modèle produit par `MIL_separe_double_encodeur_CAMELYON.py`.
- Architecture « plate » : le script attend un MIL à une branche dont les clés sont
  `encodeur_patch.*`, `attention.*`, `classifieur.*`, c'est-à-dire un modèle sauvegardé par
  `03_mil_une_echelle/train_patch_MIL.py`. Les modèles `MIL20x` du dossier 04/05
  (`modele_hopt20x.pth`, `modele_routage_20x.pth`) ont des clés `branche.*` : ils ne se chargent
  pas tels quels (`Missing key(s) in state_dict`). Pour les charger, remplacer la classe `MIL` du
  script par `BrancheAttention` + `MIL20x` copiées depuis `MIL_separe_double_encodeur.py`.
- Une seule classe : CAMELYON est 100 % FH dans ce projet. L'AUC n'est donc pas définie
  (`nan`, normal), et l'accuracy correspond au taux de FH correctement reconnus (rappel FH).

### À modifier

| Variable | Rôle |
|---|---|
| `MODELE` | chemin du `.pth` à tester |
| `ENCODEUR` / `DIM` | doivent correspondre au modèle : Virchow2 → `"virchow_v2"` / `2560` ; H-optimus-1 → `"h_optimus_1"` / `1536` |
| `MANIFEST_CAMELYON` | manifest des lames CAMELYON encodées avec ce même encodeur |
| `DOSSIER_SORTIE` | où écrire la matrice et le CSV |
| `MAX_PATCHS_PAR_LAME` | `None` = tous les patchs ; mettre un nombre si la mémoire GPU manque |
