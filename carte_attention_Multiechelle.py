# ===========================================================================
# CARTE D'ATTENTION superposee sur la lame (MIL MULTI-ECHELLE 20x + 5x)
# ---------------------------------------------------------------------------
# Le modele a DEUX branches d'attention (une par echelle). Il produit donc
# DEUX cartes : une pour le 20x (patches fins) et une pour le 5x (contexte).
# On charge les patches + coords aux deux echelles, on passe les deux dans le
# modele, et on superpose chaque carte de chaleur sur la vignette de la lame.
# ===========================================================================

from pathlib import Path
import h5py
import numpy as np
import torch
import torch.nn as nn
import openslide
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# --- CONFIG (a adapter) ---
DOSSIER_SORTIE     = Path("/mnt/e/internship/output_LF_virchow2")
DOSSIER_PATCHS_20X = Path("/mnt/e/internship/output_LF_virchow2/patches")
DOSSIER_PATCHS_5X  = Path("/mnt/e/internship/output_LF_virchow2_5x/patches")  # <-- verifier ce chemin
MODELE_PATH  = DOSSIER_SORTIE / "modele_MIL_concatenation_Virchow2.pth"
ENCODEUR     = "virchow_v2"
DIM_FEATURES = 2560
CLASSES      = ["FH", "FL"]

# La lame a visualiser : son nom (sans .h5) ET le chemin de son .svs
NOM_LAME   = ""
CHEMIN_SVS = Path(f"/mnt/d/internship/slides/{NOM_LAME}.svs")   # <-- adapter si besoin

NIVEAU_VIGNETTE = 2          # niveau de resolution pour le fond (plus haut = plus petit/rapide)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- le meme modele que l'entrainement (doit etre IDENTIQUE) ---
class BrancheAttention(nn.Module):
    def __init__(self, dim_entree):
        super().__init__()
        self.encodeur_patch = nn.Sequential(
            nn.Linear(dim_entree, 256), nn.ReLU(), nn.Dropout(0.3),
        )
        self.attention = nn.Sequential(
            nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1),
        )

    def forward(self, patches):
        h = self.encodeur_patch(patches)
        scores = self.attention(h)
        poids = torch.softmax(scores, dim=0)
        vecteur_lame = torch.sum(poids * h, dim=0)
        return vecteur_lame, poids


class MILMultiEchelle(nn.Module):
    def __init__(self, dim_entree, nb_classes):
        super().__init__()
        self.branche_20x = BrancheAttention(dim_entree)
        self.branche_5x  = BrancheAttention(dim_entree)
        self.classifieur = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, nb_classes),
        )

    def forward(self, patches_20x, patches_5x):
        v20, poids20 = self.branche_20x(patches_20x)
        v5,  poids5  = self.branche_5x(patches_5x)
        v_fusion = torch.cat([v20, v5], dim=0)
        sortie = self.classifieur(v_fusion.unsqueeze(0))
        return sortie, poids20, poids5


# --- charger features + coords d'un H5 ---
def charger_h5(h5_path):
    with h5py.File(h5_path, "r") as f:
        feats = f[f"features/{ENCODEUR}"][...]     # (N, DIM_FEATURES)
        coords = f["coords"][...]                   # (N, >=4) : x, y, w, h, [niveau]
    return feats, coords


# --- construire la carte de chaleur a l'echelle de la vignette ---
def construire_overlay(coords, poids, h_vig, w_vig, facteur):
    carte = np.zeros((h_vig, w_vig), dtype=np.float32)
    x = coords[:, 0].astype(float)                  # colonne 0 = x (coords niveau 0)
    y = coords[:, 1].astype(float)                  # colonne 1 = y (coords niveau 0)

    # taille du patch en coords natives : lue depuis les coords (w, h) si dispo.
    # -> gere automatiquement le fait qu'un patch 5x couvre plus de surface qu'un 20x.
    if coords.shape[1] >= 4:
        tw = float(np.median(coords[:, 2]))
        th = float(np.median(coords[:, 3]))
    else:
        tw = th = 512.0                             # repli si pas de w/h

    poids_norm = (poids - poids.min()) / (poids.max() - poids.min() + 1e-8)
    tvw = max(1, int(tw / facteur))                 # largeur du patch dans la vignette
    tvh = max(1, int(th / facteur))                 # hauteur du patch dans la vignette

    for xi, yi, pi in zip(x, y, poids_norm):
        xv = int(xi / facteur); yv = int(yi / facteur)
        if 0 <= yv < h_vig and 0 <= xv < w_vig:
            bloc = carte[yv:yv+tvh, xv:xv+tvw]
            carte[yv:yv+tvh, xv:xv+tvw] = np.maximum(bloc, pi)   # garde le plus fort si chevauchement

    sigma = max(tvw, tvh) / 2                        # lissage proportionnel a la taille du patch
    return gaussian_filter(carte, sigma=sigma)


def main():
    # 1. charger le modele entraine
    modele = MILMultiEchelle(DIM_FEATURES, len(CLASSES)).to(DEVICE)
    modele.load_state_dict(torch.load(MODELE_PATH, map_location=DEVICE))
    modele.eval()
    print(f"Modele charge depuis {MODELE_PATH}")

    # 2. charger les features ET les coords aux DEUX echelles
    feats20, coords20 = charger_h5(DOSSIER_PATCHS_20X / f"{NOM_LAME}.h5")
    feats5,  coords5  = charger_h5(DOSSIER_PATCHS_5X  / f"{NOM_LAME}.h5")
    print(f"20x : {len(feats20)} patches | 5x : {len(feats5)} patches")
    # sanity-check du systeme de coords : les deux max (x,y) doivent etre du meme
    # ordre de grandeur (= dimensions niveau 0). Si le 5x est ~4x plus petit,
    # c'est que ses coords sont en unites de son propre niveau -> a corriger.
    print(f"20x coords max (x,y) = ({coords20[:,0].max():.0f}, {coords20[:,1].max():.0f})")
    print(f"5x  coords max (x,y) = ({coords5[:,0].max():.0f}, {coords5[:,1].max():.0f})")

    # 3. passer les deux echelles dans le modele -> deux jeux de poids
    with torch.no_grad():
        x20 = torch.from_numpy(feats20).float().to(DEVICE)
        x5  = torch.from_numpy(feats5).float().to(DEVICE)
        sortie, poids20, poids5 = modele(x20, x5)
        proba = torch.softmax(sortie, dim=1)[0].cpu().numpy()
        poids20 = poids20.squeeze().cpu().numpy()    # (N20,)
        poids5  = poids5.squeeze().cpu().numpy()     # (N5,)
    pred = CLASSES[int(proba.argmax())]
    print(f"Prediction : {pred}  (proba FH={proba[0]:.3f}, FL={proba[1]:.3f})")

    # 4. vignette de fond (commune aux deux cartes)
    lame = openslide.OpenSlide(str(CHEMIN_SVS))
    niveau = min(NIVEAU_VIGNETTE, lame.level_count - 1)
    facteur = lame.level_downsamples[niveau]
    vignette = np.array(
        lame.read_region((0, 0), niveau, lame.level_dimensions[niveau]).convert("RGB")
    )
    h_vig, w_vig = vignette.shape[:2]
    print(f"Vignette : {w_vig}x{h_vig} (niveau {niveau}, facteur {facteur:.1f})")

    # 5. une carte par echelle
    carte20 = construire_overlay(coords20, poids20, h_vig, w_vig, facteur)
    carte5  = construire_overlay(coords5,  poids5,  h_vig, w_vig, facteur)

    # 6. affichage cote a cote (20x | 5x)
    fig, axes = plt.subplots(1, 2, figsize=(20, 11))
    for ax, carte, titre in [(axes[0], carte20, "branche 20x"),
                             (axes[1], carte5,  "branche 5x")]:
        ax.imshow(vignette)
        ax.imshow(carte, cmap="jet", alpha=0.45)     # jet : bleu=faible, rouge=fort
        ax.set_title(f"Attention {titre}", fontsize=12)
        ax.axis("off")
    fig.suptitle(f"Carte d'attention MIL multi-echelle — {NOM_LAME}\n"
                 f"encodeur : {ENCODEUR} | Prediction : {pred} "
                 f"(FH={proba[0]:.2f}, FL={proba[1]:.2f})", fontsize=13)
    plt.tight_layout()
    chemin = DOSSIER_SORTIE / f"attention_{NOM_LAME}_{ENCODEUR}.png"
    plt.savefig(chemin, dpi=150, bbox_inches="tight")
    print(f"Carte d'attention enregistree -> {chemin}")


if __name__ == "__main__":
    main()