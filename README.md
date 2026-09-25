# Stage_HF_LF : Classification FH / FL sur lames histologiques entières

Code de mon stage Aéro 4 à l'IUCT-Oncopole (2026) : classifier automatiquement des lames
numérisées (WSI, *Whole Slide Images*) entre hyperplasie folliculaire (FH, bénin) et
lymphome folliculaire (FL, malin), à partir de features extraites par des modèles de
fondation en pathologie (ResNet50, H-optimus-0, H-optimus-1, Virchow2).

La méthode a d'abord été mise au point sur un jeu de lymphomes de Hodgkin (591 lames), puis
appliquée au jeu lymphome folliculaire (651 lames). Le détail des résultats est dans le rapport
de stage.

---

## Sommaire

1. [Vue d'ensemble du pipeline](#1-vue-densemble-du-pipeline)
2. [Organisation du dépôt](#2-organisation-du-dépôt)
3. [Installation (à faire une seule fois)](#3-installation-à-faire-une-seule-fois)
4. [Données d'entrée attendues](#4-données-dentrée-attendues)
5. [Étape 1 : Patchifier et encoder avec AtlasPatch](#5-étape-1--patchifier-et-encoder-avec-atlaspatch)
6. [Étape 2 : Labelliser et découper par patient](#6-étape-2--labelliser-et-découper-par-patient)
7. [Étapes suivantes : entraîner, évaluer, interpréter](#7-étapes-suivantes--entraîner-évaluer-interpréter)
8. [Adapter les scripts à ses propres données](#8-adapter-les-scripts-à-ses-propres-données)
9. [Résultats principaux](#9-résultats-principaux)
10. [Points d'attention connus](#10-points-dattention-connus)

---

## 1. Vue d'ensemble du pipeline

```
 Lames .svs  ──►  AtlasPatch  ──►  1 fichier .h5 par lame  ──►  Label_and_split.py  ──►  manifest_slides.csv
 (WSI brutes)     (segmentation      coords + features/<encodeur>   (label + split          (1 ligne par lame :
                   SAM2, patchs                                      train/val/test          label, patient, type,
                   256×256, encodage)                                par patient)            split, chemin du .h5)
                                                                                                   │
        ┌──────────────────────────────────────────────────────────────────────────────────────────┘
        ▼
 02 Vote majoritaire  →  03 MIL 1 échelle  →  04 MIL multi-échelle  →  05 Modèle final (double encodeur)
                                                                              │
                                              06 Sélection de features   07 Cartes d'attention
```

Toutes les étapes après AtlasPatch travaillent uniquement sur les features (fichiers `.h5`)
et le manifest (CSV). Les lames `.svs` ne sont plus relues, sauf pour dessiner les cartes
d'attention (étape 07).

---

## 2. Organisation du dépôt

| Dossier | Contenu | README |
|---|---|---|
| `01_preparation/` | Labellisation, split patient, manifest 5x | [01_preparation/README.md](01_preparation/README.md) |
| `02_vote_majorite/` | MLP par patch + vote majoritaire (20x, 5x, ensemble) | [02_vote_majorite/README.md](02_vote_majorite/README.md) |
| `03_mil_une_echelle/` | MIL à attention sur une seule échelle (20x) | [03_mil_une_echelle/README.md](03_mil_une_echelle/README.md) |
| `04_mil_multiechelle/` | MIL 20x + 5x : moyenne, concaténation, routage | [04_mil_multiechelle/README.md](04_mil_multiechelle/README.md) |
| `05_Modèle final - …/` | Double encodeur (biopsie → H-optimus, pièce → Virchow2), test externe CAMELYON | [README](<05_Modèle final - Modèle multi échelle avec changement d'encodeur en fonction du type de lame/README.md>) |
| `06_selection_features/` | Quelles dimensions des features sont utiles ? | [06_selection_features/README.md](06_selection_features/README.md) |
| `07_cartes_attention/` | Cartes de chaleur de l'attention superposées sur la lame | [07_cartes_attention/README.md](07_cartes_attention/README.md) |

> Le nom du dossier `05_…` contient des espaces et des apostrophes : il faut toujours le mettre
> entre guillemets dans le terminal (`cd "05_Modèle final - …"`), ou utiliser la touche Tab.

---

## 3. Installation (à faire une seule fois)

### 3.1 Matériel et système

- Un GPU NVIDIA est fortement recommandé (le stage a été fait sur une RTX 5090 Laptop, 24 Go
  de VRAM, 64 Go de RAM). L'encodage de ~600 lames prend plusieurs heures à plusieurs jours selon
  l'encodeur.
- Linux ou WSL2/Ubuntu sous Windows (c'est ce qui a été utilisé).
- Beaucoup de place disque : les lames brutes font ~900 Go pour 591 lames ; les `.h5` de features
  sont bien plus petits mais comptent quand même plusieurs dizaines de Go par encodeur.

### 3.2 Environnement conda

Si conda n'est pas installé : installer [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

```bash
# 1. créer l'environnement (Python >= 3.10 obligatoire pour AtlasPatch)
conda create -n atlas_patch python=3.10
conda activate atlas_patch

# 2. OpenSlide (bibliothèque système pour lire les .svs), AVANT atlas-patch
conda install -c conda-forge openslide
#   (ou, hors conda : sudo apt-get install openslide-tools)

# 3. AtlasPatch + SAM2 (SAM2 sert à segmenter le tissu)
pip install atlas-patch
pip install git+https://github.com/facebookresearch/sam2.git

# 4. les encodeurs de pathologie (H-optimus, Virchow2, …) sont dans une option
pip install "atlas-patch[patch-encoders]"

# 5. le reste des dépendances du dépôt (PyTorch, scikit-learn, h5py, …)
git clone https://github.com/SimonLignac/Stage_HF_LF.git
cd Stage_HF_LF
pip install -r requirements.txt
```

Vérifier que tout marche :

```bash
atlaspatch info                                           # liste les formats supportés
python -c "import torch; print(torch.cuda.is_available())"   # doit afficher True
```

> Si `torch.cuda.is_available()` renvoie `False`, installer la version de PyTorch qui correspond
> à votre version de CUDA depuis [pytorch.org](https://pytorch.org/get-started/locally/).

### 3.3 Accès aux modèles de fondation (Hugging Face)

Les poids des encodeurs sont téléchargés automatiquement par AtlasPatch au premier lancement
(plusieurs Go, puis mis en cache dans `~/.cache/huggingface`). ResNet50 est libre d'accès, mais
H-optimus-0, H-optimus-1 et Virchow2 sont des modèles à accès restreint :

1. Créer un compte sur [huggingface.co](https://huggingface.co).
2. Demander l'accès (accepter les conditions) sur chaque page :
   - H-optimus-0 : <https://huggingface.co/bioptimus/H-optimus-0>
   - H-optimus-1 : <https://huggingface.co/bioptimus/H-optimus-1>
   - Virchow2 : <https://huggingface.co/paige-ai/Virchow2>
3. Créer un *token* (Settings → Access Tokens, droit « Read »).
4. Le donner à AtlasPatch avant de lancer l'encodage :

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx
# pour ne pas le retaper à chaque session :
echo 'export HF_TOKEN=hf_xxxxxxxxxxxxxxxxx' >> ~/.bashrc
```

Sans ce token (ou sans accès accepté), l'encodage s'arrête avec une erreur *401 / access denied*.

### 3.4 (WSL) Monter les disques externes

Les lames et les sorties étaient sur des disques USB (`D:` et `E:` côté Windows). Sous WSL, s'ils
n'apparaissent pas dans `/mnt/`, il faut les monter :

```bash
sudo mkdir -p /mnt/e
sudo mount -t drvfs E: /mnt/e
ls /mnt/e/internship/          # vérifier qu'on voit bien les fichiers
```

---

## 4. Données d'entrée attendues

### 4.1 Les lames

Un dossier contenant les lames numérisées (`.svs`, mais aussi `.tif`, `.ndpi`, `.mrxs`, …) :

```
/mnt/d/internship/slides/
├── LAME_001.svs
├── LAME_002.svs
└── ...
```

### 4.2 Le fichier de labels (CSV)

Un CSV (séparateur `,`) avec une ligne par lame. Colonnes utilisées par
`01_preparation/Label_and_split.py` :

| Colonne | Obligatoire | Contenu |
|---|---|---|
| `case_id` | oui | identifiant du patient (sert au split anti-fuite) |
| `slide_id` | oui | nom de la lame sans extension, identique au nom du `.svs` (donc du `.h5`) |
| `label` | oui | `FH` ou `FL` |
| `type` | pour 04/05 | type de prélèvement : `Biopsy` ou `Surgical specimen` |
| `cohort`, `grade`, `diagnosis` | non | recopiés tels quels dans le manifest (utiles pour analyser les erreurs) |

Exemple :

```csv
case_id,slide_id,label,cohort,type,grade,diagnosis
P001,LAME_001,FL,Oncopole,Biopsy,2,Lymphome folliculaire grade 2
P001,LAME_002,FL,Oncopole,Surgical specimen,2,Lymphome folliculaire grade 2
P002,LAME_003,FH,Oncopole,Biopsy,,Hyperplasie folliculaire
```

---

## 5. Étape 1 : Patchifier et encoder avec AtlasPatch

[AtlasPatch](https://github.com/AtlasAnalyticsLab/AtlasPatch) fait tout en une seule commande :

1. segmente le tissu sur une vignette de la lame avec SAM2 (le fond blanc est ignoré) ;
2. découpe le tissu en patchs de `--patch-size` pixels au grossissement `--target-mag` ;
3. encode chaque patch avec le ou les modèles de `--feature-extractors` ;
4. écrit un fichier `.h5` par lame dans `<output>/patches/`.

### 5.1 Commande de base

```bash
conda activate atlas_patch

atlaspatch process /mnt/d/internship/slides \
  --output /mnt/d/internship/output \
  --patch-size 256 \
  --target-mag 20 \
  --feature-extractors resnet50 \
  --device cuda \
  --patch-workers 2
```

| Option | Rôle | Valeurs utilisées |
|---|---|---|
| 1er argument | dossier des lames (ou une seule lame) | `/mnt/d/internship/slides` |
| `--output` | dossier de sortie (les `.h5` vont dans `<output>/patches/`) | un dossier par encodeur et par grossissement |
| `--patch-size` | taille des patchs en pixels | `256` |
| `--target-mag` | grossissement | `20` (détails cellulaires) ou `5` (architecture globale) |
| `--feature-extractors` | nom exact de l'encodeur | `resnet50`, `h_optimus_0`, `h_optimus_1`, `virchow_v2` |
| `--device` | `cuda` (GPU) ou `cpu` | `cuda` |
| `--feature-batch-size` | nb de patchs encodés à la fois sur le GPU | `128` (H-optimus), `64` (Virchow2, plus gros) |
| `--feature-num-workers` | processus qui préparent les patchs pour le GPU | `8` (H-optimus), `4` (Virchow2) |
| `--patch-workers` | processus qui découpent les lames | `2` à `8` |

> Si le GPU manque de mémoire (*CUDA out of memory*), baisser `--feature-batch-size`.
> Si la RAM sature, baisser `--feature-num-workers` et `--patch-workers`.

### 5.2 Les encodeurs utilisés

| Encodeur (`--feature-extractors`) | Dimension des features | Accès HF | Utilisé dans |
|---|---|---|---|
| `resnet50` (ImageNet, généraliste) | 2048 | libre | 02 (comparaison de départ) |
| `h_optimus_0` (Bioptimus) | 1536 | restreint | 02, 04 (jeu Hodgkin) |
| `h_optimus_1` (Bioptimus) | 1536 | restreint | 03, 05, 06, 07 |
| `virchow_v2` (Paige) | 2560 | restreint | 04, 05, 07 (multi-échelle) |

La dimension est à reporter dans la variable `DIM_FEATURES` (ou `*_DIM`) des scripts.
Beaucoup d'autres encodeurs sont disponibles (UNI, CONCH, Phikon, GigaPath…) : voir la liste
dans la documentation d'AtlasPatch.

### 5.3 Lancer un long encodage en arrière-plan (`nohup`)

Un encodage complet dure des heures : on le lance avec `nohup … &` pour qu'il continue même si le
terminal est fermé, et on écrit la sortie dans un fichier log.

```bash
conda activate atlas_patch

# vérifier qu'aucun AtlasPatch ne tourne déjà (n'affiche rien si c'est libre)
pgrep -f atlaspatch

nohup atlaspatch process /mnt/d/internship/slides \
  --output /mnt/e/internship/output_LF \
  --patch-size 256 --target-mag 20 \
  --feature-extractors h_optimus_1 --device cuda \
  --feature-batch-size 128 --feature-num-workers 8 --patch-workers 8 \
  > ~/encode_hoptimus.log 2>&1 &
```

Suivre l'avancement :

```bash
# dernière ligne de la barre de progression de l'encodage
tail -5 ~/encode_hoptimus.log | tr '\r' '\n' | grep "Feature embedding" | tail -1

# ou suivre le log en continu (Ctrl+C pour quitter, l'encodage continue)
tail -f ~/encode_hoptimus.log

# nombre de lames déjà traitées
ls /mnt/e/internship/output_LF/patches/*.h5 | wc -l
```

### 5.4 Enchaîner plusieurs encodages (20x et 5x)

Les modèles multi-échelle (04, 05) ont besoin des features au 20x et au 5x pour chaque lame :
il faut lancer AtlasPatch deux fois, avec deux dossiers de sortie différents. Exemple utilisé pour
CAMELYON (H-optimus-1 20x, puis Virchow2 20x, puis Virchow2 5x ; `&&` = on ne passe à l'étape
suivante que si la précédente a réussi) :

```bash
nohup bash -c '
atlaspatch process /mnt/d/internship_CAMELYON/CAMELYON \
  --output /mnt/e/internship_CAMELYON/output_hoptimus \
  --patch-size 256 --target-mag 20 \
  --feature-extractors h_optimus_1 --device cuda \
  --feature-batch-size 128 --feature-num-workers 8 --patch-workers 8 \
&& atlaspatch process /mnt/d/internship_CAMELYON/CAMELYON \
  --output /mnt/e/internship_CAMELYON/output_virchow2 \
  --patch-size 256 --target-mag 20 \
  --feature-extractors virchow_v2 --device cuda \
  --feature-batch-size 64 --feature-num-workers 4 --patch-workers 8 \
&& atlaspatch process /mnt/d/internship_CAMELYON/CAMELYON \
  --output /mnt/e/internship_CAMELYON/output_virchow2_5x \
  --patch-size 256 --target-mag 5 \
  --feature-extractors virchow_v2 --device cuda \
  --feature-batch-size 64 --feature-num-workers 4 --patch-workers 8 \
' > ~/encode_camelyon.log 2>&1 &

tail -f ~/encode_camelyon.log
```

### 5.5 Ce que produit AtlasPatch

```
/mnt/e/internship/output_LF/
└── patches/
    ├── LAME_001.h5
    ├── LAME_002.h5
    └── ...
```

Chaque `.h5` contient :

| Clé | Forme | Contenu |
|---|---|---|
| `coords` | `(N_patchs, 5)` | pour chaque patch : `x, y` (niveau 0), `largeur, hauteur` lues, niveau pyramidal |
| `features/<encodeur>` | `(N_patchs, dim)` | un vecteur de features par patch, ex. `features/h_optimus_1` → `(N, 1536)` |

Pour vérifier un fichier :

```python
import h5py
with h5py.File("/mnt/e/internship/output_LF/patches/LAME_001.h5", "r") as f:
    print(list(f.keys()), list(f["features"].keys()))
    print(f["coords"].shape, f["features/h_optimus_1"].shape)
```

> Plusieurs encodeurs peuvent cohabiter dans le même `.h5` (`features/resnet50`,
> `features/h_optimus_0`, …) si on relance AtlasPatch avec le même `--output`.
> Dans le stage, un dossier de sortie séparé a été utilisé par encodeur et par grossissement
> (`output_LF`, `output_LF_virchow2`, `output_LF_virchow2_5x`, …).

---

## 6. Étape 2 : Labelliser et découper par patient

Une fois les `.h5` produits, on associe chaque lame à son label et à un split
train / val / test par patient (jamais une lame d'un même patient dans deux splits) :

```bash
python 01_preparation/Label_and_split.py     # 20x : crée manifest_slides_*.csv et manifest_patches_*.csv
python 01_preparation/Manifest_5x.py         # 5x  : réutilise EXACTEMENT le même split que le 20x
```

Les chemins sont à modifier en haut de chaque script. Détails :
[01_preparation/README.md](01_preparation/README.md).

---

## 7. Étapes suivantes : entraîner, évaluer, interpréter

Ordre dans lequel les scripts ont été utilisés pendant le stage (tous depuis la racine du dépôt,
avec `conda activate atlas_patch`) :

```bash
# 01 : préparation
python 01_preparation/Label_and_split.py
python 01_preparation/Manifest_5x.py

# 02 : vote majoritaire (MLP par patch)
python 02_vote_majorite/train_patch_majority.py
python 02_vote_majorite/train_patch_5x.py
python 02_vote_majorite/Moyenne_5x_20x.py

# 03 : MIL à attention, une échelle
python 03_mil_une_echelle/train_patch_MIL.py

# 04 : MIL multi-échelle
python 04_mil_multiechelle/5x_20x_MIL_Moyenne.py
python 04_mil_multiechelle/MIL_train_concatenation.py
python 04_mil_multiechelle/MIL_separe.py

# 05 : modèle final (double encodeur) + test externe
python "05_Modèle final - Modèle multi échelle avec changement d'encodeur en fonction du type de lame/MIL_separe_double_encodeur.py"
python "05_Modèle final - Modèle multi échelle avec changement d'encodeur en fonction du type de lame/MIL_separe_double_encodeur_CAMELYON.py"
python "05_Modèle final - Modèle multi échelle avec changement d'encodeur en fonction du type de lame/Modele_final_TEST.py"

# 06 : sélection de features
python 06_selection_features/Selection_features.py

# 07 : cartes d'attention (après avoir sauvegardé un modèle en 03, 04 ou 05)
python 07_cartes_attention/Carte_attention.py
python 07_cartes_attention/carte_attention_Multiechelle.py
```

Pour un long entraînement, le même principe `nohup` qu'en 5.3 s'applique :

```bash
nohup python 03_mil_une_echelle/train_patch_MIL.py > ~/mil.log 2>&1 &
tail -f ~/mil.log
```

---

## 8. Adapter les scripts à ses propres données

Les scripts n'ont pas d'arguments en ligne de commande : tous les réglages sont des
constantes en MAJUSCULES en haut de chaque fichier. Pour les adapter, il faut presque toujours
modifier les mêmes éléments :

| Variable | À mettre |
|---|---|
| `MANIFEST`, `MANIFEST_20X`, `MANIFEST_5X`, `*_MANIFEST_*` | chemin du/des manifest(s) créé(s) à l'étape 2 |
| `DOSSIER_SORTIE` | où écrire les figures (courbes, matrices de confusion) et les modèles `.pth` |
| `ENCODEUR` / `*_ENCODEUR` | nom exact de l'encodeur, identique à `--feature-extractors` (clé `features/<nom>` du `.h5`) |
| `DIM_FEATURES` / `*_DIM` | dimension de cet encodeur (voir tableau 5.2) |
| `CLASSES` et `label_vers_indice` | les labels du CSV, ex. `{"FH": 0, "FL": 1}` |
| `TYPE_BIOPSIE` | libellé exact des biopsies dans la colonne `type` (`"Biopsy"`) |
| `SEED` | graine aléatoire (changer pour tester la stabilité des résultats) |
| `NB_EPOCHS`, `PATIENCE`, `TAUX_APPRENTISSAGE`, `MAX_PATCHS_PAR_LAME` | hyperparamètres d'entraînement |

Erreurs typiques :

- `KeyError: "Unable to open object (object 'xxx' doesn't exist)"` → `ENCODEUR` ne correspond pas
  à la clé `features/<nom>` du `.h5`.
- `mat1 and mat2 shapes cannot be multiplied` → `DIM_FEATURES` ne correspond pas à l'encodeur.
- `KeyError: 'XX'` sur `label_vers_indice` → le CSV contient un label non prévu dans le dictionnaire.
- `FileNotFoundError` → disque non monté (voir 3.4) ou chemin du manifest à changer.

---

## 9. Résultats principaux

Jeu lymphome folliculaire, test = 130 lames (55 FH, 75 FL) :

| Modèle | Accuracy | AUC |
|---|---|---|
| MIL une échelle (H-optimus-1, 20x) | 0,946 | 0,988 |
| MIL multi-échelle (Virchow2, 20x + 5x) | 0,931 | 0,993 |
| Double encodeur (biopsie → H-optimus-1 20x, pièce → Virchow2 20x+5x) | 0,962 | 0,989 |

Jeu Hodgkin (mise au point), test = 118 lames : de 0,746 / 0,866 (vote majoritaire ResNet50) à
0,949 / 0,984 (MIL multi-échelle H-optimus-1).

---

## 10. Points d'attention connus

- Chemins en dur : chaque script pointe vers les disques du stage (`/mnt/d/...`, `/mnt/e/...`).
  Ils sont à modifier avant toute exécution.
- Noms de manifest : `Label_and_split.py` écrit `manifest_slides_CAMELYON.csv` (dernière
  utilisation), alors que d'autres scripts lisent `manifest_slides.csv` ou `manifest_slides_LF.csv`.
  Harmoniser les noms selon le jeu de données.
- Noms de modèles `.pth` : les scripts de cartes d'attention et de test chargent un fichier
  `.pth` précis. Vérifier que le nom et l'architecture correspondent au script d'entraînement
  (voir les README de 05 et 07).
- Les limites propres à chaque script sont listées dans le README de son dossier.

## Licence

Voir [LICENSE](LICENSE). AtlasPatch est sous licence CC-BY-NC-SA-4.0, et chaque modèle de
fondation a sa propre licence (usage de recherche non commercial pour la plupart).
