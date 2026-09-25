# ===========================================================================
# CARTE D'ATTENTION superposee sur la lame (MIL)
# ---------------------------------------------------------------------------
# Montre OU le modele MIL "regarde" sur la lame pour decider FH/FL.
# On recupere le poids d'attention de chaque patch, on le place a sa position
# (x, y), et on superpose cette carte de chaleur sur la vignette de la lame.
#
# PREREQUIS : un modele entraine sauvegarde (modele_MIL.pth), cf. torch.save
# dans le script d'entrainement.
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
from matplotlib import cm
from scipy.ndimage import gaussian_filter

# --- CONFIG (a adapter) ---
DOSSIER_SORTIE = Path("/mnt/e/internship/output_LF")
DOSSIER_PATCHS = DOSSIER_SORTIE / "patches"
MODELE_PATH = DOSSIER_SORTIE / "modele_MIL.pth"
ENCODEUR     = "h_optimus_1"
DIM_FEATURES = 1536
CLASSES = ["FH", "FL"]

# La lame a visualiser : son nom (sans .h5) ET le chemin de son .svs
NOM_LAME = ""
CHEMIN_SVS = Path(f"/mnt/d/internship/slides/{NOM_LAME}.svs")   # <-- adapter si besoin

NIVEAU_VIGNETTE = 2          # niveau de resolution pour le fond (plus haut = plus petit/rapide)
TAILLE_PATCH_NATIF = 512     # taille du patch en coords natives (vu dans tes coords)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- le meme modele que l'entrainement (doit etre IDENTIQUE) ---
class MILAttention(nn.Module):
    def __init__(self, dim_entree, nb_classes):
        super().__init__()
        self.encodeur_patch = nn.Sequential(
            nn.Linear(dim_entree, 256), nn.ReLU(), nn.Dropout(0.3),
        )
        self.attention = nn.Sequential(
            nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1),
        )
        self.classifieur = nn.Linear(256, nb_classes)

    def forward(self, patches):
        h = self.encodeur_patch(patches)
        scores_attention = self.attention(h)
        poids = torch.softmax(scores_attention, dim=0)
        vecteur_lame = torch.sum(poids * h, dim=0)
        sortie = self.classifieur(vecteur_lame)
        return sortie.unsqueeze(0), poids


def main():
    # 1. charger le modele entraine
    modele = MILAttention(DIM_FEATURES, len(CLASSES)).to(DEVICE)
    modele.load_state_dict(torch.load(MODELE_PATH, map_location=DEVICE))
    modele.eval()
    print(f"Modele charge depuis {MODELE_PATH}")

    # 2. charger les features ET les coords de la lame
    h5_path = DOSSIER_PATCHS / f"{NOM_LAME}.h5"
    with h5py.File(h5_path, "r") as f:
        feats = f[f"features/{ENCODEUR}"][...]     # (N, 1536)
        coords = f["coords"][...]                   # (N, 5) : x, y, w, h, niveau
    print(f"{len(feats)} patches charges")

    # 3. passer dans le modele pour recuperer les POIDS d'attention
    with torch.no_grad():
        x = torch.from_numpy(feats).float().to(DEVICE)
        sortie, poids = modele(x)
        proba = torch.softmax(sortie, dim=1)[0].cpu().numpy()
        poids = poids.squeeze().cpu().numpy()        # (N,) : un poids par patch
    pred = CLASSES[int(proba.argmax())]
    print(f"Prediction : {pred}  (proba FH={proba[0]:.3f}, FL={proba[1]:.3f})")

    # 4. charger la vignette de la lame (fond)
    lame = openslide.OpenSlide(str(CHEMIN_SVS))
    niveau = min(NIVEAU_VIGNETTE, lame.level_count - 1)
    facteur = lame.level_downsamples[niveau]         # combien la vignette est reduite
    vignette = lame.read_region((0, 0), niveau, lame.level_dimensions[niveau]).convert("RGB")
    vignette = np.array(vignette)
    h_vig, w_vig = vignette.shape[:2]
    print(f"Vignette : {w_vig}x{h_vig} (niveau {niveau}, facteur {facteur:.1f})")

    # 5. construire la carte de chaleur a l'echelle de la vignette
    carte = np.zeros((h_vig, w_vig), dtype=np.float32)
    x = coords[:, 0].astype(float)                   # colonne 0 = x
    y = coords[:, 1].astype(float)                   # colonne 1 = y
    # normaliser les poids entre 0 et 1 pour l'affichage
    poids_norm = poids 
    taille_vig = int(TAILLE_PATCH_NATIF / facteur)   # taille d'un patch dans la vignette
    for xi, yi, pi in zip(x, y, poids_norm):
        xv = int(xi / facteur); yv = int(yi / facteur)
        if 0 <= yv < h_vig and 0 <= xv < w_vig:
            carte[yv:yv+taille_vig, xv:xv+taille_vig] = pi


    # 6. superposer la carte sur la vignette
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(vignette)
    ax.imshow(carte, cmap="jet", alpha=0.45)         # jet : bleu=faible, rouge=fort
    ax.set_title(f"Carte d'attention MIL — encodeur : {ENCODEUR}\n"
                 f"Prediction : {pred} (FH={proba[0]:.2f}, FL={proba[1]:.2f})", fontsize=11)
    ax.axis("off")
    plt.tight_layout()
    chemin = DOSSIER_SORTIE / f"attention_{NOM_LAME}_{ENCODEUR}.png"
    plt.savefig(chemin, dpi=150, bbox_inches="tight")
    print(f"Carte d'attention enregistree -> {chemin}")


if __name__ == "__main__":
    main()