# 01. Préparation : labels, split par patient, manifests

À lancer après AtlasPatch (voir le [README principal](../README.md#5-étape-1--patchifier-et-encoder-avec-atlaspatch))
et avant tout entraînement.

| Script | Rôle |
|---|---|
| [`Label_and_split.py`](Label_and_split.py) | associe chaque `.h5` à son label et à son patient, crée le split train/val/test par patient, écrit les manifests |
| [`Manifest_5x.py`](Manifest_5x.py) | crée le manifest des features 5x en recopiant le split du 20x |

---

## `Label_and_split.py`

### Ce qu'il fait

1. Lit le CSV de labels (`FICHIER_LABELS`) et liste les `.h5` présents dans `DOSSIER_PATCHS`.
2. Apparie chaque ligne du CSV à son `.h5` via `slide_id` (= nom du fichier sans `.h5`).
   Les lames du CSV sans `.h5` sont comptées et signalées (`[!] N slide_id sans H5 correspondant`),
   par exemple si l'encodage n'est pas fini ou si le nom ne correspond pas exactement.
3. Compte les patchs de chaque lame (`coords.shape[0]`).
4. Split par patient avec `StratifiedGroupKFold(n_splits=5)` :
   - `groups=case_id` : toutes les lames d'un patient tombent dans le même pli ;
   - stratifié sur `label` : chaque pli garde la proportion FH/FL ;
   - le pli de test du 1er découpage devient test (~20 %), celui du 2e découpage devient
     val (~20 %), le reste est train (~60 %).
5. Vérifie l'absence de fuite : `assert` qu'aucun patient n'est dans deux splits.
6. Écrit deux manifests dans `DOSSIER_SORTIE` :

| Fichier | Une ligne par | Colonnes | Utilisé par |
|---|---|---|---|
| `manifest_slides_*.csv` | lame | `slide, case_id, label, cohort, type, grade, diagnosis, n_patches, h5_path, split` | tous les scripts d'entraînement |
| `manifest_patches_*.csv` | patch (plusieurs millions de lignes) | `slide, case_id, label, split, patch_idx, x, y, read_w, read_h, level` | analyses au niveau patch (non utilisé par les scripts actuels) |

Le split est reproductible : avec la même `GRAINE` et le même CSV, on retrouve exactement les
mêmes splits à chaque exécution.

### Lancer

```bash
conda activate atlas_patch
python 01_preparation/Label_and_split.py
```

### À modifier

| Variable | Rôle | Valeur dans le dépôt (dernière utilisation : CAMELYON) |
|---|---|---|
| `DOSSIER_PATCHS` | dossier `patches/` produit par AtlasPatch | `/mnt/e/internship_CAMELYON/output_virchow2/patches` |
| `FICHIER_LABELS` | CSV des labels | `.../fh_dataset_CAMELYON.csv` |
| `DOSSIER_SORTIE` | où écrire les manifests | `/mnt/e/internship_CAMELYON/output_virchow2` |
| `NB_MORCEAUX` | nombre de plis (5 → 60/20/20 ; 10 → 80/10/10) | `5` |
| `GRAINE` | graine du tirage | `42` |
| `COL_*` | noms des colonnes du CSV, si les vôtres sont différents | `case_id`, `slide_id`, `label`, … |
| noms des fichiers de sortie | lignes 105-106 et 125-126 | `manifest_slides_CAMELYON.csv`, `manifest_patches_CAMELYON.csv` |

> Attention aux noms de sortie : les scripts d'entraînement lisent selon les cas
> `manifest_slides.csv` (jeu Hodgkin), `manifest_slides_LF.csv` (jeu FL) ou
> `manifest_slides_CAMELYON.csv`. Renommer la sortie (lignes 105-106) selon le jeu traité.

Si le CSV utilise un autre séparateur (`;` typique d'un export Excel français), changer
`sep=","` ligne 37. Si un nom de lame du CSV diffère du nom de fichier (extension, suffixe…),
adapter la fonction `trouver_lame` (lignes 41-44).

### Plusieurs encodeurs = même split ?

Le split ne dépend que du CSV, de la graine et de la liste des lames appariées. Si on relance
le script pour un autre encodeur (autre `DOSSIER_PATCHS`) avec les mêmes lames, on obtient
le même split. C'est indispensable en 05, où H-optimus et Virchow2 sont combinés : le script 05
vérifie d'ailleurs que les splits sont identiques et s'arrête sinon. Si une lame manque pour un
encodeur, la répartition peut changer : vérifier que tous les encodages sont complets.

---

## `Manifest_5x.py`

### Ce qu'il fait

Pour le 5x, on ne refait pas de split : on part du manifest 20x (qui a déjà les splits) et on
remplace seulement `h5_path` par le chemin du `.h5` 5x de la même lame, et `n_patches` par le
nombre de patchs au 5x (environ 15 fois moins qu'au 20x). Résultat : même split exactement, donc
une comparaison 20x / 5x honnête et la possibilité de combiner les deux échelles lame par lame.

Les lames qui n'ont pas de `.h5` au 5x sont signalées et exclues.

### Lancer

```bash
python 01_preparation/Manifest_5x.py
```

(à lancer après `Label_and_split.py` sur le 20x, et après l'encodage AtlasPatch avec
`--target-mag 5`)

### À modifier

| Variable | Rôle |
|---|---|
| `MANIFEST_20X` | manifest lames 20x produit par `Label_and_split.py` |
| `DOSSIER_5X` | dossier `patches/` de l'encodage 5x (`--target-mag 5`) |
| `MANIFEST_5X` | nom du manifest 5x à créer |

Le même encodeur doit avoir été utilisé au 20x et au 5x si on veut les concaténer dans un même
modèle (en 04/05 : Virchow2 aux deux échelles).
