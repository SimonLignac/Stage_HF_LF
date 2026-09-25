# 07 : Cartes d'attention

## Objectif

Montrer où le modèle MIL regarde sur la lame pour décider FH ou FL. Chaque patch a un poids
d'attention (calculé par le modèle) : on place ce poids à la position du patch sur la lame et on
superpose la carte de chaleur obtenue sur une vignette de la lame (bleu = peu d'attention,
rouge = forte attention). Le pathologiste peut ainsi vérifier que le modèle s'appuie sur des zones
pertinentes (follicules) et non sur des artefacts (bords, plis, marqueurs).

| Script | Modèle attendu | Résultat |
|---|---|---|
| [`Carte_attention.py`](Carte_attention.py) | MIL une échelle (dossier 03) | 1 carte |
| [`carte_attention_Multiechelle.py`](carte_attention_Multiechelle.py) | MIL concaténé 20x + 5x (dossiers 04/05) | 2 cartes côte à côte (branche 20x, branche 5x) |

## Prérequis

- Un modèle entraîné et sauvegardé (`.pth`) ;
- le `.h5` de la lame (features + `coords`), encodé avec le même encodeur que le modèle ;
- le fichier `.svs` d'origine de la lame (pour dessiner le fond), lu avec OpenSlide.

## Méthode (commune aux deux scripts)

1. Recréer l'architecture du modèle (elle doit être identique à celle de l'entraînement)
   et charger les poids.
2. Lire `features/<ENCODEUR>` et `coords` dans le `.h5` de la lame.
3. Passer tous les patchs dans le modèle → prédiction de la lame + un poids par patch.
4. Lire une vignette de la lame dans le `.svs` au niveau pyramidal `NIVEAU_VIGNETTE`.
5. Pour chaque patch, convertir `(x, y)` (coordonnées au niveau 0) en pixels de vignette
   (division par `level_downsamples[niveau]`) et remplir le carré correspondant avec son poids.
6. Superposer la carte sur la vignette (colormap `jet`, transparence 0,45) et enregistrer
   `attention_<NOM_LAME>_<ENCODEUR>.png` dans `DOSSIER_SORTIE`, avec la prédiction et les
   probabilités FH/FL dans le titre.

Différences de `carte_attention_Multiechelle.py` :
- charge les `.h5` 20x et 5x, obtient deux jeux de poids (un par branche) ;
- la taille d'un patch sur la lame est lue dans `coords` (colonnes largeur/hauteur), ce qui gère
  automatiquement le fait qu'un patch 5x couvre 4× plus de surface qu'un patch 20x ;
- les poids sont normalisés entre 0 et 1, les chevauchements gardent le poids le plus fort, et
  la carte est lissée par un filtre gaussien (rendu plus lisible) ;
- affiche les coordonnées max 20x et 5x : elles doivent être du même ordre de grandeur (sinon
  les coordonnées 5x ne sont pas au niveau 0 et la carte 5x sera décalée).

## Lancer

1. Choisir une lame (par ex. une lame mal classée affichée par les scripts d'entraînement).
2. Renseigner son nom sans extension dans `NOM_LAME` et vérifier `CHEMIN_SVS`.
3. Lancer :

```bash
conda activate atlas_patch
python 07_cartes_attention/Carte_attention.py
python 07_cartes_attention/carte_attention_Multiechelle.py
```

Une lame par exécution : pour en faire plusieurs, relancer en changeant `NOM_LAME` (ou mettre le
contenu de `main()` dans une boucle sur une liste de noms).

## À modifier

### `Carte_attention.py`

| Variable | Rôle / remarque |
|---|---|
| `DOSSIER_SORTIE` | où écrire l'image ; sert aussi à trouver `patches/` et le modèle |
| `DOSSIER_PATCHS` | dossier des `.h5` (par défaut `DOSSIER_SORTIE / "patches"`) |
| `MODELE_PATH` | `modele_MIL.pth` : ⚠ `03_mil_une_echelle/train_patch_MIL.py` sauvegarde sous le nom `modele_MIL_Virchow2.pth`, il faut renommer le fichier ou changer ce chemin |
| `ENCODEUR` / `DIM_FEATURES` | ceux du modèle (`h_optimus_1` / `1536` par défaut) |
| `NOM_LAME` | vide dans le dépôt : à remplir |
| `CHEMIN_SVS` | chemin du `.svs` (changer l'extension si `.ndpi`, `.tif`, …) |
| `NIVEAU_VIGNETTE` | niveau pyramidal du fond : plus grand = image plus petite et plus rapide |
| `TAILLE_PATCH_NATIF` | taille d'un patch en pixels au niveau 0 : 512 pour des patchs 256 px à 20x sur une lame scannée à 40x. À adapter si le scanner ou `--patch-size` change (la 5e colonne de `coords` et les colonnes 3-4 donnent l'info) |

Remarque : dans ce script les poids ne sont pas normalisés (`poids_norm = poids`). Ça ne gêne pas
l'affichage (`imshow` s'adapte au min/max), mais les zones sans tissu valent 0 et les patchs ont
des poids très petits (somme = 1). Pour un rendu comparable au multi-échelle, on peut reprendre la
normalisation min-max et le `gaussian_filter` de `carte_attention_Multiechelle.py`.

### `carte_attention_Multiechelle.py`

| Variable | Rôle / remarque |
|---|---|
| `DOSSIER_SORTIE` | où écrire l'image |
| `DOSSIER_PATCHS_20X`, `DOSSIER_PATCHS_5X` | dossiers `patches/` des encodages 20x et 5x |
| `MODELE_PATH` | modèle concaténé à 2 branches. Les fichiers compatibles produits par le dépôt sont `modele_routage_concat.pth` (04, `MIL_separe.py`) et `modele_vir_concat.pth` (05). Le nom par défaut `modele_MIL_concatenation_Virchow2.pth` n'est produit par aucun script actuel |
| `ENCODEUR` / `DIM_FEATURES` | `virchow_v2` / `2560` par défaut |
| `NOM_LAME`, `CHEMIN_SVS`, `NIVEAU_VIGNETTE` | comme ci-dessus |

> `modele_MIL_adaptatif.pth` (04, `MIL_train_concatenation.py`) a les mêmes clés de poids et se
> charge aussi. En revanche, pour une biopsie, ce modèle n'utilise pas la branche 5x : sa carte 5x
> n'a alors pas de sens.

## Erreurs fréquentes

| Erreur | Cause |
|---|---|
| `Missing key(s) / Unexpected key(s) in state_dict` | le `.pth` ne correspond pas à l'architecture du script (1 branche vs 2 branches) |
| `size mismatch for encodeur_patch.0.weight` | `DIM_FEATURES` différent de celui du modèle |
| `OpenSlideUnsupportedFormatError` / fichier introuvable | mauvais `CHEMIN_SVS`, ou disque non monté |
| carte décalée ou minuscule | `TAILLE_PATCH_NATIF` ou coordonnées 5x pas au niveau 0 |
