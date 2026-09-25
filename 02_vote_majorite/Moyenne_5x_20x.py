# ===========================================================================
# ENSEMBLE 20x + 5x (moyenne des scores) AVEC FIGURES
# ---------------------------------------------------------------------------
# Comme le script d'ensemble, mais on trace pour CHAQUE modele (20x et 5x) :
#   - ses courbes d'apprentissage (loss train, accuracy val, AUC val)
#   - sa matrice de confusion sur le test
# Plus la comparaison finale (20x / 5x / ensemble).
# ===========================================================================

from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (confusion_matrix, accuracy_score,
                             classification_report, roc_auc_score)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --- REGLAGES COMMUNS ---
DIM_FEATURES = 1536
CLASSES = ["FH", "FL"]
MAX_PATCHS_PAR_LAME = 500
NB_EPOCHS = 8
TAILLE_LOT = 1024
TAUX_APPRENTISSAGE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# Les deux manifests (meme split, echelles differentes)
MANIFEST_20X = Path("/mnt/d/internship/output/manifest_slides.csv")
MANIFEST_5X  = Path("/mnt/e/internship/output_5x/manifest_slides_5x.csv")
DOSSIER_SORTIE = Path("/mnt/d/internship/output")     # ou seront sauvees les figures
ENCODEUR = "h_optimus_0"

torch.manual_seed(SEED); np.random.seed(SEED)
label_vers_indice = {"FH": 0, "FL": 1}
generateur = np.random.default_rng(SEED)


def charger_echantillon(sous_tableau):
    X, y = [], []
    for _, r in sous_tableau.iterrows():
        with h5py.File(r["h5_path"], "r") as f:
            feats = f[f"features/{ENCODEUR}"][...]
        if len(feats) > MAX_PATCHS_PAR_LAME:
            idx = generateur.choice(len(feats), MAX_PATCHS_PAR_LAME, replace=False)
            feats = feats[idx]
        X.append(feats)
        y.append(np.full(len(feats), label_vers_indice[r["label"]]))
    return np.concatenate(X), np.concatenate(y)


class JeuDePatchs(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


class ReseauPatch(nn.Module):
    def __init__(self, dim_entree, nb_classes):
        super().__init__()
        self.reseau = nn.Sequential(
            nn.Linear(dim_entree, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, nb_classes),
        )
    def forward(self, x): return self.reseau(x)


# --- calcule les scores par lame ET la loss patch pour un groupe ---
def scores_par_lame(modele, lames, critere=None):
    modele.eval()
    scores, vrais, noms = [], [], []
    total_loss, nb_patches = 0.0, 0
    with torch.no_grad():
        for _, r in lames.iterrows():
            with h5py.File(r["h5_path"], "r") as f:
                feats = f[f"features/{ENCODEUR}"][...]
            y_lame = label_vers_indice[r["label"]]
            probas = []
            for i in range(0, len(feats), TAILLE_LOT):
                xb = torch.from_numpy(feats[i:i+TAILLE_LOT]).float().to(DEVICE)
                logits = modele(xb)
                probas.append(torch.softmax(logits, 1).cpu().numpy())
                # loss : on compare chaque patch au label de sa lame
                if critere is not None:
                    yb = torch.full((len(xb),), y_lame, dtype=torch.long, device=DEVICE)
                    total_loss += critere(logits, yb).item() * len(xb)
                    nb_patches += len(xb)
            probas = np.concatenate(probas)
            scores.append(probas[:, 1].mean())
            vrais.append(y_lame)
            noms.append(r["slide"])
    val_loss = (total_loss / nb_patches) if nb_patches > 0 else float("nan")
    return np.array(scores), np.array(vrais), noms, val_loss


# --- Entraine un modele, TRACE ses courbes, renvoie les scores test ---
def entrainer_et_scorer(manifest_path, nom):
    print(f"\n===== Modele {nom} ({manifest_path.name}) =====")
    tableau = pd.read_csv(manifest_path)

    Xtr, ytr = charger_echantillon(tableau[tableau.split == "train"])
    loader = DataLoader(JeuDePatchs(Xtr, ytr), batch_size=TAILLE_LOT, shuffle=True, num_workers=4)

    lames_val  = tableau[tableau.split == "val"].reset_index(drop=True)
    lames_test = tableau[tableau.split == "test"].reset_index(drop=True)

    modele = ReseauPatch(DIM_FEATURES, len(CLASSES)).to(DEVICE)
    opt = torch.optim.AdamW(modele.parameters(), lr=TAUX_APPRENTISSAGE, weight_decay=1e-2)
    crit = nn.CrossEntropyLoss()

    # historique pour les courbes
    historique = {"train_loss": [], "val_loss": [], "val_acc": [], "val_auc": []}

    for epoch in range(1, NB_EPOCHS + 1):
        modele.train(); total = 0.0; nb = 0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(modele(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item(); nb += 1
        train_loss = total / max(nb, 1)

        # mesures sur la validation (au niveau lame) + val_loss
        sc_val, vr_val, _, val_loss = scores_par_lame(modele, lames_val, critere=crit)
        val_acc = accuracy_score(vr_val, (sc_val >= 0.5).astype(int))
        try:    val_auc = roc_auc_score(vr_val, sc_val)
        except ValueError: val_auc = float("nan")

        historique["train_loss"].append(train_loss)
        historique["val_loss"].append(val_loss)
        historique["val_acc"].append(val_acc)
        historique["val_auc"].append(val_auc)
        print(f"  epoch {epoch}/{NB_EPOCHS}  train_loss {train_loss:.4f} | "
              f"val_loss {val_loss:.4f} | val_acc {val_acc:.3f} | val_auc {val_auc:.3f}")

    # (les courbes sont tracees a la fin, pour les deux modeles ensemble)

    # scores sur le test (pour l'ensemble ET la matrice de confusion)
    sc_test, vr_test, noms, _ = scores_par_lame(modele, lames_test)

    # --- MATRICE DE CONFUSION (une figure par modele) ---
    tracer_matrice(vr_test, (sc_test >= 0.5).astype(int), nom)

    df = pd.DataFrame({"slide": noms, "vrai": vr_test, f"score_{nom}": sc_test})
    return df, historique


# --- trace une matrice de confusion en image ---
def tracer_matrice(vrais, predits, nom):
    matrice = confusion_matrix(vrais, predits, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(matrice, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASSES); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predit"); ax.set_ylabel("Vrai")
    ax.set_title(f"Matrice de confusion ({nom})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, matrice[i, j], ha="center", va="center",
                    color="white" if matrice[i, j] > matrice.max()/2 else "black",
                    fontsize=14)
    plt.tight_layout()
    chemin = DOSSIER_SORTIE / f"matrice_confusion_{nom}.png"
    plt.savefig(chemin, dpi=130); plt.close()
    print(f"  matrice de confusion -> {chemin}")


# --- trace les courbes des DEUX modeles : 20x en haut, 5x en bas (2 lignes x 3 colonnes) ---
def tracer_courbes_combinees(hist20, hist5):
    ep = range(1, NB_EPOCHS + 1)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    for ligne, (hist, nom) in enumerate([(hist20, "20x"), (hist5, "5x")]):
        # colonne 0 : loss train vs val
        axes[ligne, 0].plot(ep, hist["train_loss"], marker="o", ms=3, label="train")
        axes[ligne, 0].plot(ep, hist["val_loss"], marker="o", ms=3, color="orange", label="validation")
        axes[ligne, 0].set_title(f"Loss train vs validation ({nom})")
        axes[ligne, 0].set_xlabel("epoch"); axes[ligne, 0].set_ylabel("loss")
        axes[ligne, 0].grid(alpha=.3); axes[ligne, 0].legend()
        # colonne 1 : accuracy validation
        axes[ligne, 1].plot(ep, hist["val_acc"], color="green", marker="o", ms=3)
        axes[ligne, 1].set_title(f"Accuracy validation ({nom})")
        axes[ligne, 1].set_xlabel("epoch"); axes[ligne, 1].set_ylabel("accuracy")
        axes[ligne, 1].grid(alpha=.3)
        # colonne 2 : AUC validation
        axes[ligne, 2].plot(ep, hist["val_auc"], color="purple", marker="o", ms=3)
        axes[ligne, 2].set_title(f"AUC validation ({nom})")
        axes[ligne, 2].set_xlabel("epoch"); axes[ligne, 2].set_ylabel("AUC")
        axes[ligne, 2].grid(alpha=.3)

    plt.tight_layout()
    chemin = DOSSIER_SORTIE / "courbes_20x_5x.png"
    plt.savefig(chemin, dpi=130); plt.close()
    print(f"\ncourbes combinees (20x en haut, 5x en bas) -> {chemin}")


def evaluer_scores(vrais, scores, titre):
    vrais = np.array(vrais); scores = np.array(scores)
    predits = (scores >= 0.5).astype(int)
    acc = accuracy_score(vrais, predits)
    auc = roc_auc_score(vrais, scores)
    print(f"\n----- {titre} -----")
    print(f"Accuracy : {acc:.3f} | AUC : {auc:.3f}")
    cm = confusion_matrix(vrais, predits, labels=[0, 1])
    print("          predit_FH  predit_FL")
    print(f"vrai_FH      {cm[0,0]:5d}    {cm[0,1]:6d}")
    print(f"vrai_FL      {cm[1,0]:5d}    {cm[1,1]:6d}")
    print(classification_report(vrais, predits, target_names=CLASSES, digits=3))
    return acc, auc


def main():
    # 1. entraine les deux modeles et recupere scores test + historiques
    df20, hist20 = entrainer_et_scorer(MANIFEST_20X, "20x")
    df5,  hist5  = entrainer_et_scorer(MANIFEST_5X,  "5x")

    # figure combinee des courbes (20x en haut, 5x en bas)
    tracer_courbes_combinees(hist20, hist5)

    # 2. on apparie par nom de lame (meme test, meme split)
    fusion = df20.merge(df5[["slide", "score_5x"]], on="slide")
    print(f"\n{len(fusion)} lames de test appariees entre 20x et 5x")

    # 3. l'ensemble = moyenne des deux scores
    fusion["score_ensemble"] = (fusion["score_20x"] + fusion["score_5x"]) / 2

    # 4. comparaison finale
    print("\n" + "="*60)
    print("COMPARAISON FINALE (sur le meme test)")
    print("="*60)
    evaluer_scores(fusion["vrai"], fusion["score_20x"],       "20x seul")
    evaluer_scores(fusion["vrai"], fusion["score_5x"],        "5x seul")
    evaluer_scores(fusion["vrai"], fusion["score_ensemble"],  "ENSEMBLE 20x+5x (moyenne)")

    # 5. matrice de confusion de l'ensemble aussi
    predits_ens = (fusion["score_ensemble"].to_numpy() >= 0.5).astype(int)
    tracer_matrice(fusion["vrai"].to_numpy(), predits_ens, "ensemble")


if __name__ == "__main__":
    main()