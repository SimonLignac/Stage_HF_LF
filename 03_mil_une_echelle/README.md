# 03 : MIL à attention, une seule échelle

Script : [`train_patch_MIL.py`](train_patch_MIL.py)

## Principe

En Multiple Instance Learning (MIL), une lame est un sac de patchs et le modèle ne reçoit
que le label de la lame, pas de label par patch. Un mécanisme d'attention donne un poids
d'importance à chaque patch : la lame est représentée par la moyenne pondérée de ses patchs,
et les patchs les plus informatifs (zones tumorales, par ex.) pèsent le plus dans la décision.

## Le modèle `MILAttention`

```
patchs (N, DIM) ──► encodeur_patch : Linear 256 + ReLU + Dropout 0.3 ──► h (N, 256)
                                              │
                         attention : Linear 128 + Tanh + Linear 1 ──► softmax sur les N patchs ──► poids (N, 1)
                                              │
                         vecteur_lame = Σ poids × h  (256) ──► classifieur : Linear 2 ──► FH / FL
```

Le modèle renvoie aussi les poids d'attention, qui servent à dessiner les cartes
d'attention (dossier 07).

## Déroulement du script

1. Lecture du manifest ; les lames sont séparées en train / val / test.
2. Entraînement lame par lame (une lame = un exemple), ordre mélangé à chaque epoch ;
   2000 patchs maximum par lame (`MAX_PATCHS_PAR_LAME`), tirés au hasard.
3. À chaque epoch, sur la validation : loss, accuracy, AUC.
4. Learning rate adaptatif (`ReduceLROnPlateau`) : lr divisé par 2 si la loss de validation
   ne baisse plus pendant 2 epochs.
5. Meilleure epoch retenue selon l'accuracy de validation (l'AUC départage les égalités) ;
   early stopping après `PATIENCE` epochs sans amélioration.
6. Rechargement des poids de la meilleure epoch, puis évaluation sur le test avec la liste
   des lames mal classées (nom, vrai label, prédiction, score FL).
7. Sauvegardes dans `DOSSIER_SORTIE` :
   - `courbes_MIL.png` (loss train/val, accuracy val, AUC val ; trait gris = meilleure epoch) ;
   - `matrice_confusion_MIL.png` ;
   - le modèle `modele_MIL_Virchow2.pth` (poids de la meilleure epoch).

## Lancer

```bash
conda activate atlas_patch
python 03_mil_une_echelle/train_patch_MIL.py
# ou en arrière-plan :
nohup python 03_mil_une_echelle/train_patch_MIL.py > ~/mil.log 2>&1 &
```

Compter quelques minutes par epoch (lecture de ~400 `.h5` par epoch).

## À modifier

| Variable | Valeur dans le dépôt | Remarque |
|---|---|---|
| `DOSSIER_SORTIE` | `/mnt/e/internship/output_LF` | figures + modèle |
| `MANIFEST` | `DOSSIER_SORTIE / "manifest_slides_LF.csv"` | |
| `ENCODEUR` / `DIM_FEATURES` | `h_optimus_1` / `1536` | `virchow_v2` / `2560`, `resnet50` / `2048`, … |
| `MAX_PATCHS_PAR_LAME` | 2000 | plus = plus d'info mais plus lent et plus de mémoire GPU |
| `NB_EPOCHS` / `PATIENCE` | 14 / 14 | avec `PATIENCE = NB_EPOCHS`, l'early stopping ne se déclenche jamais ; mettre par ex. `NB_EPOCHS = 50`, `PATIENCE = 10` |
| `TAUX_APPRENTISSAGE` | `1e-4` | |
| `SEED` | 82 | changer pour tester la stabilité |
| nom du `.pth` (ligne 250) | `modele_MIL_Virchow2.pth` | ⚠ le nom est fixe quel que soit l'encodeur : le renommer (ex. `modele_MIL_{ENCODEUR}.pth`) pour ne pas confondre/écraser les modèles |

> Le modèle sauvegardé ici a une architecture « plate » (`encodeur_patch.*`, `attention.*`,
> `classifieur.*`). C'est celle qu'attendent `07_cartes_attention/Carte_attention.py` (qui cherche
> `modele_MIL.pth`) et `05_…/Modele_final_TEST.py` : il faut renommer le fichier ou changer le
> chemin dans ces scripts.

## Résultats

| Jeu | Encodeur | Accuracy | AUC |
|---|---|---|---|
| Hodgkin (test = 118) | H-optimus-1, 20x | 0,932 | 0,977 |
| Lymphome folliculaire (test = 130) | H-optimus-1, 20x | 0,946 | 0,988 |

Par rapport au vote majoritaire (0,864 / 0,956 sur Hodgkin), le MIL améliore nettement les deux
métriques.
