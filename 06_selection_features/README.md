# 06 : Sélection de features

Script : [`Selection_features.py`](Selection_features.py)

## Objectif

Chaque patch est décrit par H-optimus-1 avec un vecteur de 1536 dimensions. La question
posée : toutes ces dimensions servent-elles à distinguer FH de FL, ou l'information est-elle
concentrée dans quelques-unes ?

Si quelques dizaines de dimensions suffisent, on peut envisager :
- des modèles plus légers et plus rapides (réduction de dimension) ;
- un premier pas vers l'interprétabilité (quelles dimensions « regarde » le modèle ?).

Cette analyse est volontairement plus simple que le MIL : elle travaille au niveau lame, avec
des outils classiques de scikit-learn (pas de réseau de neurones, pas de GPU).

---

## Prérequis

- Les `.h5` de la lame encodés avec l'encodeur choisi (étape AtlasPatch, voir le
  [README principal](../README.md#5-étape-1--patchifier-et-encoder-avec-atlaspatch)).
- Le manifest lames créé par [`01_preparation/Label_and_split.py`](../01_preparation/README.md),
  avec au minimum les colonnes `h5_path`, `label`, `case_id`, `split`.

Pas besoin de GPU : le script tourne en quelques minutes sur CPU. Le plus long est la lecture des
`.h5` (une fois par lame).

---

## Lancer le script

```bash
conda activate atlas_patch
cd ~/Stage_HF_LF
python 06_selection_features/Selection_features.py
```

Le script affiche ses résultats dans le terminal et écrit 3 fichiers dans `DOSSIER_SORTIE`.

---

## Ce que fait le code, étape par étape

### Configuration (lignes 32-38)

```python
MANIFEST = Path("/mnt/e/internship/output_LF/manifest_slides_LF.csv")
ENCODEUR = "h_optimus_1"
DIM_FEATURES = 1536
DOSSIER_SORTIE = Path("/mnt/e/internship/output_LF")
label_vers_indice = {"FH": 0, "FL": 1}
SEED = 42
```

### Fonction `vecteur_lame_moyen(h5_path)` (lignes 41-45)

Ouvre le `.h5` d'une lame, lit la matrice `features/<ENCODEUR>` de forme `(N_patchs, 1536)` et
renvoie la moyenne sur tous les patchs, soit un seul vecteur de 1536 valeurs pour la lame.

C'est la façon la plus simple de passer de « un sac de patchs » à « un tableau classique »
(une ligne = une lame, une colonne = une dimension), ce que les outils scikit-learn savent traiter.
Contrairement au MIL, tous les patchs sont utilisés (pas de plafond `MAX_PATCHS_PAR_LAME`).

### Étape 1 : Construire la matrice `X` (lignes 52-60)

Pour chaque lame du manifest :
- `X` reçoit son vecteur moyen → matrice `(nb_lames, 1536)` (651 × 1536 pour le jeu FL) ;
- `y` reçoit son label converti en entier (`FH → 0`, `FL → 1`) ;
- `groupes` reçoit son `case_id` (patient), utilisé plus bas pour la validation croisée.

### Restriction au TRAIN (lignes 62-65)

Seules les lames `split == "train"` sont gardées (`Xtr`, `ytr`, `gtr`). Les lames de val et de
test ne sont jamais utilisées pour choisir les features : sinon on choisirait des dimensions
« qui marchent sur le test », et le score mesuré serait artificiellement bon (fuite de données).

### Étape 2 : Standardisation (lignes 67-69)

`StandardScaler` recentre chaque dimension (moyenne 0) et la met à l'échelle (variance 1), en
calculant moyenne et écart-type sur le train uniquement. Sans ça, une dimension qui varie
naturellement sur de grandes valeurs dominerait les autres dans le test F et la régression.

### Étape 3a. Méthode A : `SelectKBest` + test F (lignes 71-78)

```python
selecteur = SelectKBest(score_func=f_classif, k=50).fit(Xtr_std, ytr)
```

Pour chaque dimension prise seule, le test F (ANOVA) compare sa moyenne chez les FH et chez
les FL par rapport à sa dispersion. Plus le score F est grand, plus la dimension sépare bien les
deux classes à elle seule. Le script affiche les 20 dimensions au meilleur score (indice + score F).

- Avantage : très rapide, facile à interpréter.
- Limite : méthode *univariée*. Elle ne voit pas les combinaisons de dimensions et garde des
  dimensions redondantes (deux dimensions très corrélées auront toutes les deux un bon score).

> Le `k=50` n'a ici aucun effet sur l'affichage : on lit directement `selecteur.scores_`, qui
> contient le score de toutes les dimensions.

### Étape 3b. Méthode B : régression logistique L1 / Lasso (lignes 80-90)

```python
logreg = LogisticRegression(penalty="l1", solver="liblinear", C=0.1, random_state=SEED)
```

On entraîne une régression logistique FH/FL avec une pénalité L1, qui pousse les coefficients
inutiles exactement à zéro. Les dimensions dont le coefficient reste non nul sont celles que
le modèle a « choisi » de garder, en tenant compte des autres (méthode *multivariée* : si deux
dimensions disent la même chose, souvent une seule est gardée).

Le script affiche :
- le nombre de dimensions gardées (coefficient ≠ 0) sur 1536 ;
- les 20 plus importantes, classées par `|coefficient|`.

`C` règle la sévérité : plus `C` est petit, plus la pénalité est forte, moins il reste de
dimensions. Avec `C=0.1`, 40 dimensions sur 1536 ont été gardées pendant le stage.

### Étape 4 : Un petit sous-ensemble suffit-il ? (lignes 92-109)

Pour `k` dans `[10, 25, 50, 100, 200, 500, 1536]`, on construit un `Pipeline` scikit-learn :

1. standardisation (`StandardScaler`) ;
2. si `k < 1536`, sélection des `k` meilleures dimensions selon le test F (`SelectKBest`) ;
3. régression logistique.

On estime l'AUC de ce pipeline par validation croisée à 5 plis, groupée par patient
(`StratifiedGroupKFold` : un patient n'est jamais à la fois dans les plis d'entraînement et
d'évaluation, et la proportion FH/FL est conservée dans chaque pli), puis on affiche l'AUC
moyenne ± l'écart-type sur les 5 plis.

Le `Pipeline` est important : à chaque tour de la validation croisée, la standardisation et le
choix des `k` dimensions sont refaits uniquement sur les 4 plis d'entraînement. Le pli
d'évaluation ne sert jamais à choisir les dimensions, donc l'AUC mesurée n'est pas biaisée
(pas de fuite de données).

Si l'AUC avec 50 dimensions est proche de celle avec 1536, alors les 1486 autres n'apportent
presque rien pour cette tâche.

### Figures et sauvegarde (lignes 111-142)

| Fichier écrit dans `DOSSIER_SORTIE` | Contenu |
|---|---|
| `selection_features_auc.png` | AUC moyenne en fonction de `k` (axe x logarithmique, axe y zoomé) |
| `selection_features_auc_0_1.png` | même courbe avec l'axe AUC fixé entre 0 et 1 (pour montrer que l'écart est faible) |
| `top_features_kbest.npy` | indices des 100 meilleures dimensions selon le test F (tableau numpy) |

Réutiliser les indices sauvegardés dans un autre script :

```python
import numpy as np
top = np.load("/mnt/e/internship/output_LF/top_features_kbest.npy")   # (100,)
feats_reduites = feats[:, top[:50]]          # garder les 50 meilleures dimensions d'une lame
```

---

## Ce qu'il faut modifier pour l'adapter

| Besoin | Où / quoi modifier |
|---|---|
| Autre jeu de données | `MANIFEST` (chemin du manifest lames) et `DOSSIER_SORTIE` |
| Autre encodeur | `ENCODEUR` (ex. `"virchow_v2"`) et `DIM_FEATURES` (ex. `2560`) : `DIM_FEATURES` sert aussi de dernière valeur de `valeurs_k` (« toutes les features ») |
| Autres labels | `label_vers_indice`, ex. `{"FH": 0, "Hodgkin": 1}` (le script est binaire : 2 classes) |
| Plus / moins de dimensions affichées | le `[:20]` des lignes 75 et 86 |
| Garder plus / moins de dimensions avec L1 | `C=0.1` ligne 82 (`C=0.01` → moins de dimensions, `C=1` → plus) |
| Tester d'autres valeurs de `k` | la liste `valeurs_k` ligne 96 |
| Nombre de plis de la validation croisée | `n_splits=5` ligne 95 |
| Sauvegarder plus de 100 indices | `[:100]` ligne 141 |
| Titre des figures | les `plt.title(...)` lignes 117 et 132 (le titre mentionne « h_optimus ») |
| Autre agrégation des patchs que la moyenne | la fonction `vecteur_lame_moyen` (ex. `feats.max(axis=0)`, ou concaténer moyenne et écart-type) |

---

## Résultats obtenus (jeu lymphome folliculaire, H-optimus-1)

Chiffres du rapport, obtenus avec la première version du script (sélection faite avant la
validation croisée, voir « Limites » plus bas). À mettre à jour après avoir relancé le script.

| Nombre de features | AUC moyenne (validation croisée par patient) |
|---|---|
| 10 | 0,945 ± 0,007 |
| 25 | 0,976 ± 0,013 |
| 50 | 0,975 ± 0,009 |
| 100 | 0,982 ± 0,008 |
| 200 | 0,983 ± 0,006 |
| 500 | 0,985 ± 0,008 |
| 1536 (toutes) | 0,982 ± 0,009 |

- Avec 10 dimensions sur 1536 (< 1 %), l'AUC atteint déjà 0,945.
- À partir d'une centaine de dimensions, on rejoint le score obtenu avec toutes les dimensions.
- La régression L1 ne garde que 40 dimensions non nulles.

L'information utile pour séparer FH et FL est donc concentrée dans une petite fraction des
dimensions de H-optimus-1.

---

## Limites et pistes d'amélioration

- Fuite corrigée dans l'étape 4. La première version du script ajustait le `StandardScaler` et
  le `SelectKBest` sur tout le train avant la validation croisée : les dimensions étaient donc
  choisies en voyant aussi les lames du pli d'évaluation, ce qui pouvait surestimer un peu l'AUC,
  surtout pour les petits `k`. La version actuelle refait la sélection dans chaque pli avec un
  `Pipeline`. Les résultats ci-dessus ont été obtenus avec l'ancienne version et sont à mettre à
  jour en relançant le script.

- Pas d'évaluation sur le test. Le script ne mesure que la validation croisée sur le train.
  Pour confirmer, il faudrait entraîner sur le train avec les `k` dimensions retenues et évaluer
  une seule fois sur les lames `split == "test"`.
- Moyenne des patchs = perte d'information. Une lame avec quelques zones tumorales dans
  beaucoup de tissu normal aura un vecteur moyen proche d'une lame bénigne. C'est pour ça que le
  MIL (03-05) reste le modèle de classification ; ici la moyenne sert juste à analyser les dimensions.
- Prochaine étape naturelle : réentraîner le MIL (03) sur `feats[:, top[:k]]` et vérifier si
  les performances se maintiennent avec un modèle beaucoup plus petit.
