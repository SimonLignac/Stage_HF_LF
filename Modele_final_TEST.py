# ===========================================================================
# INFERENCE EXTERNE : CAMELYON en TEST uniquement
# Modele MIL SIMPLE (une seule branche attention, architecture PLATE)
# ---------------------------------------------------------------------------
# On charge UN modele deja entraine (sur LF, SANS CAMELYON) et on predit sur
# les lames CAMELYON. Aucun entrainement.
#
# ATTENTION 1 (anti-fuite) : le .pth ne doit PAS avoir vu CAMELYON.
# ATTENTION 2 : CAMELYON = 100% FH -> une seule classe. AUC = nan (normal),
#   accuracy = rappel sur FH.
# ===========================================================================

from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES = ["FH", "FL"]
MAX_PATCHS_PAR_LAME = None     # None = tous les patchs (test 100% reproductible)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 10034

# ------------------- LE modele a tester -------------------
MODELE = Path("/mnt/e/internship/output_LF_virchow2/modele_MIL.pth")             # a verifier
# Adapte ENCODEUR + DIM + MANIFEST au modele charge :
#   Virchow2  -> ENCODEUR = "virchow_v2",  DIM = 2560
#   H-optimus -> ENCODEUR = "h_optimus_1", DIM = 1536
ENCODEUR = "virchow_v2"
DIM = 2560
MANIFEST_CAMELYON = Path("/mnt/e/internship_CAMELYON/output_virchow2/manifest_slides_CAMELYON.csv")
DOSSIER_SORTIE    = Path("/mnt/e/internship_CAMELYON/output_virchow2")

torch.manual_seed(SEED); np.random.seed(SEED)
label_vers_indice = {"FH": 0, "FL": 1}
generateur = np.random.default_rng(SEED)


def charger_lame(h5_path, encodeur):
    with h5py.File(h5_path, "r") as f:
        feats = f[f"features/{encodeur}"][...]
    if MAX_PATCHS_PAR_LAME is not None and len(feats) > MAX_PATCHS_PAR_LAME:
        idx = generateur.choice(len(feats), MAX_PATCHS_PAR_LAME, replace=False)
        feats = feats[idx]
    return feats


# ---- Architecture PLATE : identique au modele sauvegarde ----
# (pas de sous-module 'branche' : les couches sont directement sur le modele,
#  ce qui reproduit les cles encodeur_patch.* / attention.* / classifieur.*)
class MIL(nn.Module):
    def __init__(self, dim_entree, nb_classes):
        super().__init__()
        self.encodeur_patch = nn.Sequential(nn.Linear(dim_entree, 256), nn.ReLU(), nn.Dropout(0.3))
        self.attention = nn.Sequential(nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1))
        self.classifieur = nn.Linear(256, nb_classes)
    def forward(self, patches):
        h = self.encodeur_patch(patches)
        poids = torch.softmax(self.attention(h), dim=0)
        v = torch.sum(poids * h, dim=0)
        return self.classifieur(v).unsqueeze(0), poids


def rapport(nom, fichier, vrais, predits, scores, noms, types, diags):
    vrais, predits, scores = np.array(vrais), np.array(predits), np.array(scores)
    print(f"\n--- Lames mal classees ({nom}) ---")
    for i in range(len(vrais)):
        if vrais[i] != predits[i]:
            print(f"  {noms[i]} : vrai={CLASSES[vrais[i]]}, predit={CLASSES[predits[i]]}, "
                  f"score_FL={scores[i]:.3f}, type={types[i]}, diag={diags[i]}")
    acc = accuracy_score(vrais, predits)
    try:
        auc = roc_auc_score(vrais, scores)
    except ValueError:
        auc = float("nan")   # une seule classe (CAMELYON = tout FH)
    print(f"\n===== {nom} =====")
    print(f"Accuracy : {acc:.3f} | AUC : {auc:.3f}   "
          f"(CAMELYON = 100% FH -> acc = rappel sur FH, AUC non definie)")
    m = confusion_matrix(vrais, predits, labels=[0, 1])
    print("          predit_FH  predit_FL")
    print(f"vrai_FH      {m[0,0]:5d}    {m[0,1]:6d}")
    print(f"vrai_FL      {m[1,0]:5d}    {m[1,1]:6d}")
    print(classification_report(vrais, predits, labels=[0, 1], target_names=CLASSES,
                                digits=3, zero_division=0))
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(m, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1]); ax.set_xticklabels(CLASSES); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predit"); ax.set_ylabel("Vrai"); ax.set_title(f"Matrice - {nom}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, m[i,j], ha="center", va="center",
                    color="white" if m[i,j] > m.max()/2 else "black", fontsize=14)
    plt.tight_layout(); plt.savefig(DOSSIER_SORTIE / fichier, dpi=130); plt.close()
    print(f"Matrice -> {DOSSIER_SORTIE / fichier}")
    return acc, auc


def main():
    DOSSIER_SORTIE.mkdir(parents=True, exist_ok=True)

    man = pd.read_csv(MANIFEST_CAMELYON)
    print(f"{len(man)} lames CAMELYON (toutes en test)")
    print(f"Labels : {man['label'].value_counts().to_dict()}")
    print(f"Types  : {man['type'].value_counts().to_dict()}")

    modele = MIL(DIM, len(CLASSES)).to(DEVICE)
    modele.load_state_dict(torch.load(MODELE, map_location=DEVICE))
    modele.eval()
    print(f"Modele charge <- {MODELE}  (encodeur={ENCODEUR}, dim={DIM})")

    vrais, predits, scores, noms, types, diags = [], [], [], [], [], []
    export = []
    with torch.no_grad():
        for _, r in man.iterrows():
            x = torch.from_numpy(charger_lame(r["h5_path"], ENCODEUR)).float().to(DEVICE)
            p = torch.softmax(modele(x)[0], 1)[0].cpu().numpy()
            vrais.append(label_vers_indice[r["label"]])
            predits.append(int(p.argmax()))
            scores.append(float(p[1]))
            noms.append(r["slide"])
            types.append(r.get("type", ""))
            diags.append(r.get("diagnosis", ""))
            export.append({"slide": r["slide"], "vrai": r["label"],
                           "type": r.get("type", ""), "predit": CLASSES[int(p.argmax())],
                           "score_FL": float(p[1])})

    acc, auc = rapport("MIL sur CAMELYON", "matrice_camelyon_MIL.png",
                       vrais, predits, scores, noms, types, diags)

    df = pd.DataFrame(export)
    chemin_csv = DOSSIER_SORTIE / "predictions_camelyon.csv"
    df.to_csv(chemin_csv, index=False)
    print(f"\nPredictions par lame -> {chemin_csv}")


if __name__ == "__main__":
    main()

