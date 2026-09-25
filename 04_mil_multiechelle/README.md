# 04 : MIL multi-échelle (20x + 5x)

Idée : le 20x montre les détails cellulaires, le 5x montre l'architecture globale du
ganglion (répartition et forme des follicules). Combiner les deux devrait aider le modèle.

Prérequis : features 20x et 5x du même encodeur, et les deux manifests avec le même split
(`01_preparation/Label_and_split.py` puis `01_preparation/Manifest_5x.py`).

| Script | Stratégie | Encodeur configuré |
|---|---|---|
| [`5x_20x_MIL.py`](5x_20x_MIL.py) | 2 MIL indépendants (20x, 5x), moyenne des scores, sans figures | H-optimus-0 |
| [`5x_20x_MIL_Moyenne.py`](5x_20x_MIL_Moyenne.py) | idem + courbes et matrices de confusion | H-optimus-0 |
| [`MIL_train_concatenation.py`](MIL_train_concatenation.py) | 1 modèle, 2 branches concaténées, 5x ignoré pour les biopsies | Virchow2 |
| [`MIL_separe.py`](MIL_separe.py) | 2 modèles + routage : biopsie → MIL 20x, pièce → MIL concaténé | Virchow2 |

---

## Brique commune : `BrancheAttention`

C'est le MIL à attention du dossier 03 sans la couche de classification : il transforme un sac
de patchs en un vecteur de lame de 256 valeurs (+ les poids d'attention).

```
patchs (N, DIM) → Linear 256 + ReLU + Dropout → attention (Linear 128 + Tanh + Linear 1) → softmax → Σ poids × h → (256)
```

Le modèle concaténé utilise une branche par échelle :

```
patchs 20x ──► BrancheAttention 20x ──► v20 (256) ─┐
                                                   ├─ concat (512) ─► Linear 128 + ReLU + Dropout ─► Linear 2
patchs 5x  ──► BrancheAttention 5x  ──► v5  (256) ─┘
```

Tous les scripts de ce dossier entraînent lame par lame, avec `AdamW` (lr `1e-4`), 2000 patchs
max par lame. Les deux derniers ajoutent `ReduceLROnPlateau`, la sélection de la meilleure epoch
(accuracy val, puis AUC) et l'early stopping, comme en 03.

---

## `5x_20x_MIL.py` et `5x_20x_MIL_Moyenne.py` : moyenne de deux MIL

1. Entraîne un `MILAttention` (identique au 03) sur le manifest 20x, puis un autre sur le 5x.
2. Pour chaque lame de test : score FL du 20x, score FL du 5x, ensemble = moyenne.
3. Compare 20x seul / 5x seul / ensemble (accuracy, AUC, matrice, rapport de classification).

`5x_20x_MIL_Moyenne.py` suit en plus la validation à chaque epoch et écrit :
`courbes_MIL_20x_5x.png`, `matrice_confusion_MIL_20x.png`, `matrice_confusion_MIL_5x.png`,
`matrice_confusion_MIL_ensemble.png`. `5x_20x_MIL.py` est la version sans validation ni figures.
Aucun des deux ne sauvegarde de modèle, et c'est la dernière epoch qui est évaluée.

À modifier : `MANIFEST_20X`, `MANIFEST_5X`, `DOSSIER_SORTIE`, `ENCODEUR`, `DIM_FEATURES`,
`NB_EPOCHS` (12 et 8).

---

## `MIL_train_concatenation.py` : un modèle adaptatif selon le type de prélèvement

Sur une biopsie (petit prélèvement), l'architecture globale vue au 5x n'apporte rien et ajoute
du bruit ; sur une pièce chirurgicale, elle peut aider (idée du Dr Franchet). Ici, un seul
modèle à deux branches :

- `type == "Biopsy"` → le vecteur 5x est remplacé par des zéros (seul le 20x compte) ;
- sinon (pièce) → les deux branches sont utilisées.

Déroulement : fusion des manifests 20x et 5x (lames communes aux deux), entraînement avec
early stopping, courbes (`courbes_MIL_adaptatif.png`), test avec lames mal classées, matrice
(`matrice_confusion_MIL_adaptatif.png`), modèle `modele_MIL_adaptatif.pth`.

À modifier : `DOSSIER_SORTIE`, `MANIFEST_20X`, `MANIFEST_5X`, `ENCODEUR`, `DIM_FEATURES`,
`NB_EPOCHS` (30), `PATIENCE` (6), `SEED`, `TYPE_BIOPSIE` (libellé exact dans la colonne `type`).

> La liste des lames mal classées sur le test affiche, pour chaque erreur, le nom de la lame, le
> vrai label, la prédiction, le score FL, le type de prélèvement et le diagnostic (colonne
> `diagnosis` du manifest, qui doit donc exister).
>
> Bug corrigé : une version précédente utilisait une variable `diags` jamais créée dans
> `metriques_lame`. L'évaluation sur le test plantait (`NameError: name 'diags' is not defined`)
> dès qu'une lame était mal classée, avant la matrice de confusion et la sauvegarde du modèle.

---

## `MIL_separe.py` : deux modèles indépendants + routage

Au lieu d'un modèle qui « éteint » le 5x, on entraîne deux modèles séparés, chacun sur
toutes les lames du train :

- `MIL20x` : une seule branche (20x) ;
- `MILConcat` : deux branches 20x + 5x concaténées.

Au moment du test, chaque lame est routée selon son type :
biopsie → prédiction de `MIL20x` ; pièce chirurgicale → prédiction de `MILConcat`.
Les biopsies profitent ainsi d'un vrai modèle 20x, dont les poids ne sont pas partagés avec le 5x.

Sorties : lames mal classées (avec type et diagnostic), `matrice_confusion_routage.png`,
`modele_routage_20x.pth`, `modele_routage_concat.pth`.

À modifier : `DOSSIER_SORTIE`, `MANIFEST_20X`, `MANIFEST_5X`, `ENCODEUR`, `DIM_FEATURES`,
`NB_EPOCHS` (50), `PATIENCE` (18), `SEED`, `TYPE_BIOPSIE`.

> `modele_routage_concat.pth` a exactement l'architecture attendue par
> `07_cartes_attention/carte_attention_Multiechelle.py` : on peut l'utiliser pour les cartes
> d'attention multi-échelle.

Cette idée est reprise et améliorée en 05, avec un encodeur différent pour chaque type de lame.

---

## Lancer

```bash
conda activate atlas_patch
python 04_mil_multiechelle/5x_20x_MIL_Moyenne.py
python 04_mil_multiechelle/MIL_train_concatenation.py
python 04_mil_multiechelle/MIL_separe.py
```

## Résultats

| Jeu | Modèle | Accuracy | AUC |
|---|---|---|---|
| Hodgkin (test = 118) | MIL concaténation 20x + 5x, H-optimus-1 | 0,949 | 0,984 |
| Lymphome folliculaire (test = 130) | MIL multi-échelle, Virchow2 | 0,931 | 0,993 |

Virchow2 a été choisi pour le multi-échelle car il a été pré-entraîné sur plusieurs
grossissements, alors que H-optimus-1 l'a surtout été au 20x.
