"""
Cree le manifest pour le 5x VIRCHOW 2 en REUTILISANT le split du 20x Virchow 2.

Principe :
  - On part du manifest_slides_LF.csv du 20x Virchow 2 (qui a deja les splits).
  - On garde tout pareil SAUF le chemin h5_path, qu'on fait pointer vers le 5x.
  - Resultat : meme split que le 20x, features 5x. Comparaison juste garantie.
"""

from pathlib import Path
import h5py
import pandas as pd

# Chemins VIRCHOW 2
MANIFEST_20X = Path("/mnt/e/internship_CAMELYON/output_virchow2/manifest_slides_CAMELYON.csv")       # manifest 20x Virchow2
DOSSIER_5X   = Path("/mnt/e/internship_CAMELYON/output_virchow2_5x/patches")                   # H5 du 5x Virchow2
MANIFEST_5X  = Path("/mnt/e/internship_CAMELYON/output_virchow2_5x/manifest_slides_5x_CAMELYON.csv")    # nouveau manifest 5x

# On lit le manifest 20x (il a deja les bons splits)
tableau = pd.read_csv(MANIFEST_20X)
print(f"{len(tableau)} lames dans le manifest 20x Virchow2")

lignes = []
manquantes = []
for _, r in tableau.iterrows():
    nouveau_h5 = DOSSIER_5X / f"{r['slide']}.h5"
    if not nouveau_h5.exists():
        manquantes.append(r["slide"])
        continue
    with h5py.File(nouveau_h5, "r") as f:
        n_patches = int(f["coords"].shape[0])
    lignes.append({
        "slide":     r["slide"],
        "case_id":   r["case_id"],
        "label":     r["label"],
        "cohort":    r.get("cohort", ""),
        "type":      r.get("type", ""),
        "grade":     r.get("grade", ""),          # copie du grade depuis le 20x
        "diagnosis": r.get("diagnosis", ""),      # copie du diagnostic depuis le 20x
        "n_patches": n_patches,           # nombre de patches au 5x (plus petit)
        "h5_path":   str(nouveau_h5),     # chemin vers le H5 du 5x Virchow2
        "split":     r["split"],          # meme split que le 20x
    })

tableau_5x = pd.DataFrame(lignes)
tableau_5x.to_csv(MANIFEST_5X, index=False)

print(f"{len(tableau_5x)} lames dans le manifest 5x Virchow2")
print(f"Splits : {tableau_5x['split'].value_counts().to_dict()}")
print(f"Patches 5x au total : {tableau_5x['n_patches'].sum()}")
print(f"(pour comparaison, 20x avait : {tableau['n_patches'].sum()} patches)")
if manquantes:
    print(f"[!] {len(manquantes)} lames 5x manquantes : {manquantes[:5]}")
print(f"-> {MANIFEST_5X}")