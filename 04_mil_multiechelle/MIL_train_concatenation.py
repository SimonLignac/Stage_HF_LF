# ===========================================================================
# MIL MULTI-ECHELLE ADAPTATIF selon le type de prelevement
# ---------------------------------------------------------------------------
# Idee (suggeree par le Dr Franchet) : sur les BIOPSIES, l'echelle 5x n'apporte
# pas d'information utile (l'architecture globale compte peu, le diagnostic est
# dans les details du 20x) et ne fait qu'ajouter du bruit. Sur les PIECES
# CHIRURGICALES, l'architecture globale (visible en 5x) peut aider.
#
# Donc :
#   - lame = "Biopsy"            -> on n'utilise QUE la branche 20x
#   - lame = "Surgical specimen" -> on utilise les DEUX branches (20x + 5x)
#
# Technique : pour une biopsie, on remplace le vecteur 5x par des zeros avant
# la concatenation. Le classifieur recoit donc toujours un vecteur de 512, mais
# pour une biopsie la moitie "5x" est neutralisee (aucune info du 5x).
# ===========================================================================

from pathlib import Path
import copy
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             classification_report, roc_auc_score)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# config

DOSSIER_SORTIE = Path("/mnt/e/internship/output_LF_virchow2")
MANIFEST_20X = Path("/mnt/e/internship/output_LF_virchow2/manifest_slides_LF.csv")
MANIFEST_5X  = Path("/mnt/e/internship/output_LF_virchow2_5x/manifest_slides_5x.csv")
ENCODEUR = "virchow_v2"
DIM_FEATURES = 2560
CLASSES = ["FH", "FL"]
MAX_PATCHS_PAR_LAME = 2000
NB_EPOCHS = 30
PATIENCE = 6
TAUX_APPRENTISSAGE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 25

# le libelle EXACT des biopsies dans la colonne "type" du manifest
TYPE_BIOPSIE = "Biopsy"

torch.manual_seed(SEED); np.random.seed(SEED)
label_vers_indice = {"FH": 0, "FL": 1}
generateur = np.random.default_rng(SEED)


def charger_lame(h5_path):
    with h5py.File(h5_path, "r") as f:
        feats = f[f"features/{ENCODEUR}"][...]
    if len(feats) > MAX_PATCHS_PAR_LAME:
        idx = generateur.choice(len(feats), MAX_PATCHS_PAR_LAME, replace=False)
        feats = feats[idx]
    return feats


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


class MILMultiEchelleAdaptatif(nn.Module):
    def __init__(self, dim_entree, nb_classes):
        super().__init__()
        self.branche_20x = BrancheAttention(dim_entree)
        self.branche_5x  = BrancheAttention(dim_entree)
        self.classifieur = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, nb_classes),
        )

    def forward(self, patches_20x, patches_5x, est_biopsie):
        # branche 20x : TOUJOURS utilisee
        v20, poids20 = self.branche_20x(patches_20x)

        if est_biopsie:
            # BIOPSIE : on ignore le 5x -> vecteur 5x mis a zero
            v5 = torch.zeros_like(v20)
            poids5 = None
        else:
            # PIECE CHIRURGICALE : on utilise le 5x normalement
            v5, poids5 = self.branche_5x(patches_5x)

        v_fusion = torch.cat([v20, v5], dim=0)          # (512,)
        sortie = self.classifieur(v_fusion.unsqueeze(0))  # (1, 2)
        return sortie, poids20, poids5


def main():
    t20 = pd.read_csv(MANIFEST_20X).set_index("slide")
    t5  = pd.read_csv(MANIFEST_5X).set_index("slide")
    communes = t20.index.intersection(t5.index)
    print(f"{len(communes)} lames communes aux deux echelles")

    lignes = []
    for slide in communes:
        lignes.append({   
            "slide": slide,
            "label": t20.loc[slide, "label"],
            "split": t20.loc[slide, "split"],
            "type":  t20.loc[slide, "type"],
            "diagnosis": t20.loc[slide, "diagnosis"],  
            "h5_20x": t20.loc[slide, "h5_path"],
            "h5_5x":  t5.loc[slide, "h5_path"],
        })
    tableau = pd.DataFrame(lignes)
    # compte biopsies / pieces dans le jeu
    print(f"Types : {tableau['type'].value_counts().to_dict()}")
    print(f"Splits : {tableau['split'].value_counts().to_dict()}")

    lames_train = tableau[tableau.split == "train"].reset_index(drop=True)
    lames_val   = tableau[tableau.split == "val"].reset_index(drop=True)

    modele = MILMultiEchelleAdaptatif(DIM_FEATURES, len(CLASSES)).to(DEVICE)
    optimiseur = torch.optim.AdamW(modele.parameters(), lr=TAUX_APPRENTISSAGE, weight_decay=1e-2)
    critere = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiseur, mode="min", factor=0.5, patience=2, min_lr=1e-6
    )

    meilleur_score = (-1.0, -1.0)
    meilleur_state = copy.deepcopy(modele.state_dict())
    meilleure_epoch = 0
    sans_amelioration = 0
    historique = {"train_loss": [], "val_loss": [], "val_acc": [], "val_auc": []}

    @torch.no_grad()
    def metriques_lame(sous_tableau, afficher_erreurs=False):
        modele.eval()
        vrais, predits, scores, noms, types, diags = [], [], [], [], [], []
        total_loss = 0.0
        for _, r in sous_tableau.iterrows():
            est_biopsie = (r["type"] == TYPE_BIOPSIE)
            feats20 = charger_lame(r["h5_20x"])
            x20 = torch.from_numpy(feats20).float().to(DEVICE)
            # pour une biopsie on ne charge meme pas le 5x (inutile) : tenseur vide
            if est_biopsie:
                x5 = torch.zeros((1, DIM_FEATURES)).to(DEVICE)   # non utilise
            else:
                feats5 = charger_lame(r["h5_5x"])
                x5 = torch.from_numpy(feats5).float().to(DEVICE)
            y = torch.tensor([label_vers_indice[r["label"]]]).to(DEVICE)
            sortie, _, _ = modele(x20, x5, est_biopsie)
            total_loss += critere(sortie, y).item()
            proba = torch.softmax(sortie, dim=1)[0].cpu().numpy()
            vrais.append(label_vers_indice[r["label"]])
            predits.append(int(proba.argmax()))
            scores.append(proba[1])
            noms.append(r["slide"]); types.append(r["type"]); diags.append(r["diagnosis"])
        vrais = np.array(vrais); predits = np.array(predits); scores = np.array(scores)

        if afficher_erreurs:
            print("\n--- Lames mal classees ---")
            for i in range(len(vrais)):
                if vrais[i] != predits[i]:
                    print(f"  {noms[i]} : vrai={CLASSES[vrais[i]]}, predit={CLASSES[predits[i]]}, "
                          f"score_FL={scores[i]:.3f}, type={types[i]}, diag={diags[i]}")

        acc = accuracy_score(vrais, predits)
        try:    auc = roc_auc_score(vrais, scores)
        except ValueError: auc = float("nan")
        loss = total_loss / len(sous_tableau)
        return acc, auc, loss, vrais, predits, scores

    print("\nEntrainement MIL multi-echelle ADAPTATIF (biopsie -> 20x seul) :")
    for epoch in range(1, NB_EPOCHS + 1):
        modele.train(); total = 0.0
        ordre = generateur.permutation(len(lames_train))
        for i in ordre:
            r = lames_train.iloc[i]
            est_biopsie = (r["type"] == TYPE_BIOPSIE)
            feats20 = charger_lame(r["h5_20x"])
            x20 = torch.from_numpy(feats20).float().to(DEVICE)
            if est_biopsie:
                x5 = torch.zeros((1, DIM_FEATURES)).to(DEVICE)
            else:
                feats5 = charger_lame(r["h5_5x"])
                x5 = torch.from_numpy(feats5).float().to(DEVICE)
            y = torch.tensor([label_vers_indice[r["label"]]]).to(DEVICE)

            optimiseur.zero_grad()
            sortie, _, _ = modele(x20, x5, est_biopsie)
            loss = critere(sortie, y)
            loss.backward()
            optimiseur.step()
            total += loss.item()
        train_loss = total / len(lames_train)

        val_acc, val_auc, val_loss, *_ = metriques_lame(lames_val)
        historique["train_loss"].append(train_loss)
        historique["val_loss"].append(val_loss)
        historique["val_acc"].append(val_acc)
        historique["val_auc"].append(val_auc)
        scheduler.step(val_loss)
        lr_actuel = optimiseur.param_groups[0]["lr"]

        auc_comparable = 0.0 if np.isnan(val_auc) else val_auc
        if (val_acc, auc_comparable) > meilleur_score:
            meilleur_score = (val_acc, auc_comparable)
            meilleur_state = copy.deepcopy(modele.state_dict())
            meilleure_epoch = epoch
            sans_amelioration = 0
        else:
            sans_amelioration += 1

        print(f"  epoch {epoch:2d}/{NB_EPOCHS}  train_loss {train_loss:.4f} | "
              f"val_loss {val_loss:.4f} | val_acc {val_acc:.3f} | "
              f"val_auc {val_auc:.3f} | lr {lr_actuel:.1e} | "
              f"patience {sans_amelioration}/{PATIENCE}")

        if sans_amelioration >= PATIENCE:
            print(f"  arret anticipe (meilleure = epoch {meilleure_epoch})")
            break

    modele.load_state_dict(meilleur_state)
    print(f"\nMeilleure epoch retenue : {meilleure_epoch} "
          f"(val_acc {meilleur_score[0]:.3f}, val_auc {meilleur_score[1]:.3f})")

    # courbes
    nb_faites = len(historique["train_loss"])
    ep = range(1, nb_faites + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(ep, historique["train_loss"], marker="o", ms=3, label="train")
    axes[0].plot(ep, historique["val_loss"], marker="o", ms=3, color="orange", label="validation")
    axes[0].axvline(meilleure_epoch, color="grey", ls="--", alpha=.6)
    axes[0].set_title("Loss train vs validation (MIL adaptatif)")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].grid(alpha=.3); axes[0].legend()
    axes[1].plot(ep, historique["val_acc"], color="green", marker="o", ms=3)
    axes[1].set_title("Accuracy validation"); axes[1].set_xlabel("epoch"); axes[1].grid(alpha=.3)
    axes[2].plot(ep, historique["val_auc"], color="purple", marker="o", ms=3)
    axes[2].set_title("AUC validation"); axes[2].set_xlabel("epoch"); axes[2].grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(DOSSIER_SORTIE / "courbes_MIL_adaptatif.png", dpi=130)
    print(f"\nCourbes -> {DOSSIER_SORTIE / 'courbes_MIL_adaptatif.png'}")

    # test
    lames_test = tableau[tableau.split == "test"].reset_index(drop=True)
    acc, auc, test_loss, vrais, predits, scores = metriques_lame(lames_test, afficher_erreurs=True)
    print(f"\n===== TEST (MIL multi-echelle ADAPTATIF) =====")
    print(f"Accuracy : {acc:.3f} | AUC : {auc:.3f} | loss : {test_loss:.4f}")
    matrice = confusion_matrix(vrais, predits, labels=[0, 1])
    print("          predit_FH  predit_FL")
    print(f"vrai_FH      {matrice[0,0]:5d}    {matrice[0,1]:6d}")
    print(f"vrai_FL      {matrice[1,0]:5d}    {matrice[1,1]:6d}")
    print(classification_report(vrais, predits, target_names=CLASSES, digits=3))

    # matrice image
    fig2, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(matrice, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASSES); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predit"); ax.set_ylabel("Vrai")
    ax.set_title("Matrice de confusion (TEST, MIL adaptatif)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, matrice[i, j], ha="center", va="center",
                    color="white" if matrice[i, j] > matrice.max()/2 else "black", fontsize=14)
    plt.tight_layout()
    plt.savefig(DOSSIER_SORTIE / "matrice_confusion_MIL_adaptatif.png", dpi=130)
    print(f"Matrice -> {DOSSIER_SORTIE / 'matrice_confusion_MIL_adaptatif.png'}")

    torch.save(modele.state_dict(), DOSSIER_SORTIE / "modele_MIL_adaptatif.pth")
    print(f"Modele sauvegarde -> {DOSSIER_SORTIE / 'modele_MIL_adaptatif.pth'}")


if __name__ == "__main__":
    main()