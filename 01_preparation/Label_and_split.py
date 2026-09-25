"""
Labellisation des patches et split par patient (anti-fuite de donnees).


Principe  :
  1. Chaque lame a un diagnostic (label) et appartient a un patient (case_id).
  2. On splitte train / val / test PAR patient (jamais par lame ni par patch).
  3. On propage ensuite le label et le split de la lame a tous ses patches.
"""

import re
from pathlib import Path

import h5py
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold



# Config
DOSSIER_PATCHS = Path("/mnt/e/internship_CAMELYON/output_virchow2/patches")
FICHIER_LABELS = Path("/mnt/c/Users/Anapath/Desktop/Stage/fh_dataset_CAMELYON.csv")
DOSSIER_SORTIE = Path("/mnt/e/internship_CAMELYON/output_virchow2")
NB_MORCEAUX = 5        # 5 folds -> on coupe en 5 morceaux qui contiennent 20 pourcents des patchs chacunes ensuite on prend 3 morceaux pour le train, 1 morceau pour le test et un morceau pour la valid
GRAINE = 42

# Colonnes (noms exacts du CSV, en minuscules)
COL_CASE   = "case_id"
COL_SLIDE  = "slide_id"
COL_LABEL  = "label"
COL_COHORT = "cohort"
COL_TYPE   = "type"
COL_GRADE     = "grade"
COL_DIAGNOSIS = "diagnosis"

# Lecture du fichier de labels
labels = pd.read_csv(FICHIER_LABELS, sep=",") #tableau panda pour les labels
labels[COL_CASE]  = labels[COL_CASE].astype(str) #converti en chaîne de caractères
nom_lames_patchs = {p.stem for p in DOSSIER_PATCHS.glob("*.h5")} # le .stem c'est le nom sans l'extension ---> permet d'enelever le .h5
#lien entre csv et les fichiers dees patchs H5 ligne par ligne : focntion qui premet de renvoyer l'id si dans les deux fichiers
def trouver_lame(slide_id, nom_lames_patchs):
    if slide_id in nom_lames_patchs:
        return slide_id
    return None

lignes, non_appariees = [], [] # lignes = lames bien mariées et l'autre c'est les celibs
for _, r in labels.iterrows(): #renvoie le _ pour ignorer l'index de la ligne on veut juste la ligne elle même
    stem = trouver_lame(r[COL_SLIDE], nom_lames_patchs) #permet de renvoyer le nom de la lame
    if stem is None:
        non_appariees.append(r[COL_SLIDE])
        continue # pour la sécurite
    with h5py.File(DOSSIER_PATCHS / f"{stem}.h5", "r") as f:#on compte les patchs
        nb_patchs = int(f["coords"].shape[0])#accède au dataset et donne le nombre de patchs
    lignes.append({
        "slide":     stem,
        "case_id":   r[COL_CASE],
        "label":     r[COL_LABEL],
        "cohort":    r.get(COL_COHORT, ""),
        "type":      r.get(COL_TYPE, ""),
        "grade":     r.get(COL_GRADE, ""),
        "diagnosis": r.get(COL_DIAGNOSIS, ""),
        "n_patches": nb_patchs,
        "h5_path":   str(DOSSIER_PATCHS / f"{stem}.h5"),
    })
# lignes est un dictionnaire donc on le convertit en tableau
tableau_lames = pd.DataFrame(lignes)
# len tableau_lames = nombre de lignes
print(f"{len(tableau_lames)} lames appariees | {tableau_lames['case_id'].nunique()} patients | "
      f"{tableau_lames['n_patches'].sum()} patches")
if non_appariees:
    print(f"[!] {len(non_appariees)} slide_id sans H5 correspondant "
          f"(ex : {non_appariees[:3]})")


#  Split par patient (anti-fuite), stratifie par label
# on decoupe en prenant e ncompte le groupe du nom du patient
# decoupeur est l'outil on l'initailise
#StratifiedGroupKFold coupe en 5 paquets, sans jamais separer un patient
# (groups=case_id, anti-fuite) et en gardant les proportions FH/FL (stratified).
# Il renvoie 5 morceaux, chacun etant un couple (train, test) :
#   morceaux[i][0] = train du morceau i  |  morceaux[i][1] = test du morceau i
# On n'en utilise que 2 : le test du morceau 0 pour notre TEST, celui du
# morceau 1 pour notre VAL, le reste en TRAIN. -> split ~60/20/20
decoupeur = StratifiedGroupKFold(n_splits=NB_MORCEAUX, shuffle=True, random_state=GRAINE)
morceaux = list(decoupeur.split(tableau_lames, tableau_lames["label"], groups=tableau_lames["case_id"]))
# Au départ, toutes les lames sont en "train"
tableau_lames["split"] = "train"

# morceaux[0] et morceaux[1] contiennent chacun (indices_train, indices_test).
# On ne garde que la 2e partie (les indices de test) de chacun.
indices_test = morceaux[0][1]   # les lignes qui iront en test
indices_val  = morceaux[1][1]   # les lignes qui iront en validation

# On change leur étiquette de "train" vers "test" et "val"
tableau_lames.loc[tableau_lames.index[indices_test], "split"] = "test"
tableau_lames.loc[tableau_lames.index[indices_val],  "split"] = "val"


#  Verification anti-fuite : un patient = un seul split
verification = tableau_lames.groupby("case_id")["split"].nunique()
fuites = verification[verification > 1].index.tolist()
assert not fuites, f"FUITE : patients presents sur plusieurs splits : {fuites}"
print("\nOK : aucun patient partage entre train / val / test")

tableau_lames.to_csv(DOSSIER_SORTIE / "manifest_slides_CAMELYON.csv", index=False)
print(f"-> {DOSSIER_SORTIE / 'manifest_slides_CAMELYON.csv'}")

# Manifest patch-level (label + split propages)
# Donc là on utilise l'apporhce du vote à majorité on regarde au niveau des patchs
lignes_patchs = []
#la liste des patchs et en gros on créée une liste et on donne le nom de la lame à chaque patch avec leur coordonnées
for index, lame in tableau_lames.iterrows():
    with h5py.File(lame["h5_path"], "r") as f:
        coords = f["coords"][...]
    for i, (x, y, rw, rh, lv) in enumerate(coords):
        lignes_patchs.append({
            "slide": lame["slide"], "case_id": lame["case_id"],
            "label": lame["label"], "split": lame["split"],
            "patch_idx": i, "x": int(x), "y": int(y),
            "read_w": int(rw), "read_h": int(rh), "level": int(lv),
        })
# on convertit le dicionnaire en tableau
tableau_patchs = pd.DataFrame(lignes_patchs)
# conversion en format csv index false pck on a pas besoin du nom de chaque ligne
tableau_patchs.to_csv(DOSSIER_SORTIE / "manifest_patches_CAMELYON.csv", index=False)
print(f"{len(tableau_patchs)} patches labellises -> {DOSSIER_SORTIE / 'manifest_patches_CAMELYON.csv'}")
#donc en gros manifest slides a 591 lignes et dans chaque ligne on a slide , case id, label, cohort, type, n patches, h5 path, et split( contient l'info train val et test). Sert à l'entrainement 
# manifest patches a environ 5 million de lignes avec slide, case id , label, split, patch idx , x, y , read_w , read_h et level. Sera utile pour le MIL pas utile pour l'instant !!
# features stockées autre part dans un fichier h5 