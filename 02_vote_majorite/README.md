# 02 : Vote majoritaire (classification patch par patch)

Première approche, la plus simple : on apprend à classer chaque patch comme FH ou FL (chaque
patch reçoit le label de sa lame), puis on décide pour la lame par vote majoritaire des patchs.

| Script | Rôle |
|---|---|
| [`train_patch_majority.py`](train_patch_majority.py) | MLP par patch au 20x, vote majoritaire (configuré pour ResNet50) |
| [`train_patch_5x.py`](train_patch_5x.py) | même chose au 5x (configuré pour H-optimus-0) |
| [`Moyenne_5x_20x.py`](Moyenne_5x_20x.py) | entraîne un modèle 20x et un modèle 5x, puis fait la moyenne de leurs scores |

---

## Le modèle commun : `ReseauPatch`

Un petit réseau de neurones (MLP) qui prend un vecteur de features et sort 2 scores (FH, FL) :

```
features (DIM_FEATURES) → Linear 512 → ReLU → Dropout 0.3 → Linear 128 → ReLU → Dropout 0.3 → Linear 2
```

- Entraînement : `AdamW` (lr `1e-4`, weight decay `1e-2`), `CrossEntropyLoss`.
- Pour limiter la mémoire, 500 patchs maximum par lame (`MAX_PATCHS_PAR_LAME`) sont tirés au
  hasard pour l'entraînement. L'évaluation, elle, utilise tous les patchs.

## Passer du patch à la lame

Pour chaque lame de val/test, on prédit tous ses patchs, puis :
- prédiction de la lame = classe majoritaire parmi les patchs (`np.bincount(...).argmax()`) ;
- score de la lame (pour l'AUC) = moyenne de la probabilité FL sur tous les patchs.

---

## `train_patch_majority.py` et `train_patch_5x.py`

Les deux scripts sont identiques, seule la configuration change (manifest 20x/5x, encodeur,
nombre d'epochs, dossier de sortie).

### Déroulement

1. Lecture du manifest, chargement des patchs train et val (500 max par lame).
2. Entraînement pendant `NB_EPOCHS` ; à chaque epoch : loss train, loss val (niveau patch),
   accuracy et AUC val (niveau lame, vote majoritaire).
3. Figure `courbes_apprentissage.png` : loss train/val, accuracy val, AUC val.
4. Évaluation finale sur le test : accuracy, AUC, matrice de confusion, rapport de classification
   (précision, rappel, F1) + `matrice_confusion_test.png`.

Aucun modèle n'est sauvegardé : c'est le modèle de la dernière epoch qui est évalué sur le test.

### Lancer

```bash
conda activate atlas_patch
python 02_vote_majorite/train_patch_majority.py
python 02_vote_majorite/train_patch_5x.py
```

### À modifier

| Variable | `train_patch_majority.py` | `train_patch_5x.py` |
|---|---|---|
| `DOSSIER_SORTIE` | `/mnt/d/internship/output` | `/mnt/d/internship/output` |
| `MANIFEST` | `DOSSIER_SORTIE / "manifest_slides.csv"` | `/mnt/e/internship/output_5x/manifest_slides_5x.csv` |
| `ENCODEUR` / `DIM_FEATURES` | `resnet50` / `2048` | `h_optimus_0` / `1536` |
| `NB_EPOCHS` | 8 | 20 |
| `BATCH_SIZE` | 8192 patchs | 8192 patchs |

- Pour comparer les encodeurs (ResNet50 vs H-optimus-0), changer uniquement `ENCODEUR` et
  `DIM_FEATURES`.
- ⚠ Les deux scripts écrivent les mêmes noms de figures dans le même dossier : lancer l'un
  après l'autre écrase les figures du premier. Changer `DOSSIER_SORTIE` ou les noms de fichiers
  (`courbes_apprentissage.png`, `matrice_confusion_test.png`) pour les garder.
- `num_workers=12` dans les `DataLoader` : à réduire si la machine a moins de cœurs.
- Si la RAM sature au chargement, baisser `MAX_PATCHS_PAR_LAME`.

---

## `Moyenne_5x_20x.py` : ensemble 20x + 5x

### Déroulement

1. Entraîne un `ReseauPatch` sur le manifest 20x, puis un second sur le manifest 5x
   (même split, cf. `01_preparation/Manifest_5x.py`).
2. Pour chaque lame de test, récupère le score FL de chaque modèle (moyenne des probabilités des
   patchs) et les apparie par nom de lame.
3. Score d'ensemble = (score 20x + score 5x) / 2, lame prédite FL si ≥ 0,5.
4. Affiche la comparaison 20x seul / 5x seul / ensemble sur le même test.

Ici la décision de la lame se fait par seuil sur la moyenne des probabilités (et non par vote
majoritaire strict comme dans les deux scripts précédents).

### Sorties (dans `DOSSIER_SORTIE`)

- `courbes_20x_5x.png` : courbes des deux modèles (20x en haut, 5x en bas) ;
- `matrice_confusion_20x.png`, `matrice_confusion_5x.png`, `matrice_confusion_ensemble.png`.

### Lancer

```bash
python 02_vote_majorite/Moyenne_5x_20x.py
```

### À modifier

`MANIFEST_20X`, `MANIFEST_5X`, `DOSSIER_SORTIE`, `ENCODEUR`, `DIM_FEATURES` (le même
encodeur est utilisé aux deux échelles), `NB_EPOCHS` (8), `TAILLE_LOT` (1024).

---

## Résultats (jeu Hodgkin, test = 118 lames)

| Stratégie | Accuracy | AUC |
|---|---|---|
| Vote majoritaire ResNet50, 20x | 0,746 | 0,866 |
| Vote majoritaire H-optimus-0, 5x | 0,856 | 0,931 |
| Vote majoritaire H-optimus-0, 20x | 0,864 | 0,956 |

Le passage à un encodeur de pathologie (H-optimus-0) améliore fortement les résultats. La moyenne
20x + 5x n'apporte pas de gain, car le modèle 5x est moins bon que le 20x.

## Limite de l'approche

Chaque patch reçoit le label de sa lame : dans une lame de lymphome, beaucoup de patchs (tissu
normal, fond, artefacts) sont donc étiquetés « FL » à tort. Le MIL (dossier 03) corrige ce
problème en laissant le modèle choisir lui-même les patchs importants.
