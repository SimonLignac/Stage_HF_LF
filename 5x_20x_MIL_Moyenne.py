# ===========================================================================
# ENSEMBLE MIL : 20x + 5x (multi-echelle avec attention) + FIGURES
# ---------------------------------------------------------------------------
# On entraine DEUX modeles MIL a attention (un sur 20x, un sur 5x),
# on recupere le score de chaque lame de test pour chacun, et on MOYENNE.
# On trace :
#   - une figure combinee des courbes (20x en haut, 5x en bas)
#   - une matrice de confusion par modele (20x, 5x, ensemble)
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

# --- REGLAGES COMMUNS ---
DIM_FEATURES = 1536
CLASSES = ["FH", "FL"]
MAX_PATCHS_PAR_LAME = 2000    # en MIL on peut en garder plus (l'attention fait le tri)
NB_EPOCHS = 8              # MIL : plus d'epochs car peu de lames (355)
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


# --- charger UNE lame entiere (tous ses patches) ---
def charger_lame(h5_path):
    with h5py.File(h5_path, "r") as f:
        feats = f[f"features/{ENCODEUR}"][...]
    if len(feats) > MAX_PATCHS_PAR_LAME:
        idx = generateur.choice(len(feats), MAX_PATCHS_PAR_LAME, replace=False)
        feats = feats[idx]
    return feats


# --- le modele MIL avec attention ---
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


# --- evalue un groupe de lames : accuracy, AUC, loss (niveau lame) + scores ---
@torch.no_grad()
def evaluer_groupe(modele, lames, critere):
    modele.eval()
    vrais, scores = [], []
    total_loss = 0.0
    for _, r in lames.iterrows():
        feats = charger_lame(r["h5_path"])
        x = torch.from_numpy(feats).float().to(DEVICE)
        y = torch.tensor([label_vers_indice[r["label"]]]).to(DEVICE)
        sortie, _ = modele(x)
        total_loss += critere(sortie, y).item()      # une loss par lame
        proba = torch.softmax(sortie, dim=1)[0].cpu().numpy()
        vrais.append(label_vers_indice[r["label"]])
        scores.append(proba[1])
    vrais = np.array(vrais); scores = np.array(scores)
    acc = accuracy_score(vrais, (scores >= 0.5).astype(int))
    try:    auc = roc_auc_score(vrais, scores)
    except ValueError: auc = float("nan")
    loss = total_loss / len(lames)
    return acc, auc, loss, vrais, scores


# --- entraine un MIL, renvoie les scores test ET l'historique ---
def entrainer_et_scorer_MIL(manifest_path, nom):
    print(f"\n===== Modele MIL {nom} ({manifest_path.name}) =====")
    tableau = pd.read_csv(manifest_path)
    lames_train = tableau[tableau.split == "train"].reset_index(drop=True)
    lames_val   = tableau[tableau.split == "val"].reset_index(drop=True)

    modele = MILAttention(DIM_FEATURES, len(CLASSES)).to(DEVICE)
    opt = torch.optim.AdamW(modele.parameters(), lr=TAUX_APPRENTISSAGE, weight_decay=1e-2)
    crit = nn.CrossEntropyLoss()

    historique = {"train_loss": [], "val_loss": [], "val_acc": [], "val_auc": []}

    for epoch in range(1, NB_EPOCHS + 1):
        modele.train()
        ordre = generateur.permutation(len(lames_train))
        total_train = 0.0
        for i in ordre:
            r = lames_train.iloc[i]
            feats = charger_lame(r["h5_path"])
            x = torch.from_numpy(feats).float().to(DEVICE)
            y = torch.tensor([label_vers_indice[r["label"]]]).to(DEVICE)
            opt.zero_grad()
            sortie, _ = modele(x)
            loss = crit(sortie, y)
            loss.backward()
            opt.step()
            total_train += loss.item()
        train_loss = total_train / len(lames_train)

        val_acc, val_auc, val_loss, *_ = evaluer_groupe(modele, lames_val, crit)
        historique["train_loss"].append(train_loss)
        historique["val_loss"].append(val_loss)
        historique["val_acc"].append(val_acc)
        historique["val_auc"].append(val_auc)
        print(f"  epoch {epoch:2d}/{NB_EPOCHS}  train_loss {train_loss:.4f} | "
              f"val_loss {val_loss:.4f} | val_acc {val_acc:.3f} | val_auc {val_auc:.3f}")

    # scores sur le test (pour l'ensemble ET la matrice de confusion)
    _, _, _, vr_test, sc_test = evaluer_groupe(modele, tableau[tableau.split == "test"].reset_index(drop=True), crit)
    noms = tableau[tableau.split == "test"]["slide"].tolist()

    # matrice de confusion de ce modele
    tracer_matrice(vr_test, (sc_test >= 0.5).astype(int), nom)

    df = pd.DataFrame({"slide": noms, "vrai": vr_test, f"score_{nom}": sc_test})
    return df, historique


# --- trace les courbes des DEUX modeles : 20x en haut, 5x en bas ---
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
    chemin = DOSSIER_SORTIE / "courbes_MIL_20x_5x.png"
    plt.savefig(chemin, dpi=130); plt.close()
    print(f"\ncourbes combinees (20x en haut, 5x en bas) -> {chemin}")


# --- trace une matrice de confusion en image ---
def tracer_matrice(vrais, predits, nom):
    matrice = confusion_matrix(vrais, predits, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(matrice, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASSES); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predit"); ax.set_ylabel("Vrai")
    ax.set_title(f"Matrice de confusion MIL ({nom})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, matrice[i, j], ha="center", va="center",
                    color="white" if matrice[i, j] > matrice.max()/2 else "black",
                    fontsize=14)
    plt.tight_layout()
    chemin = DOSSIER_SORTIE / f"matrice_confusion_MIL_{nom}.png"
    plt.savefig(chemin, dpi=130); plt.close()
    print(f"  matrice de confusion -> {chemin}")


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
    # 1. entraine les deux MIL, recupere scores test + historiques
    df20, hist20 = entrainer_et_scorer_MIL(MANIFEST_20X, "20x")
    df5,  hist5  = entrainer_et_scorer_MIL(MANIFEST_5X,  "5x")

    # figure combinee des courbes (20x en haut, 5x en bas)
    tracer_courbes_combinees(hist20, hist5)

    # 2. on apparie par nom de lame (meme test, meme split)
    fusion = df20.merge(df5[["slide", "score_5x"]], on="slide")
    print(f"\n{len(fusion)} lames de test appariees entre 20x et 5x")

    # 3. l'ensemble = moyenne des deux scores
    fusion["score_ensemble"] = (fusion["score_20x"] + fusion["score_5x"]) / 2

    # 4. comparaison finale
    print("\n" + "="*60)
    print("COMPARAISON FINALE MIL (sur le meme test)")
    print("="*60)
    evaluer_scores(fusion["vrai"], fusion["score_20x"],       "MIL 20x seul")
    evaluer_scores(fusion["vrai"], fusion["score_5x"],        "MIL 5x seul")
    evaluer_scores(fusion["vrai"], fusion["score_ensemble"],  "ENSEMBLE MIL 20x+5x (moyenne)")

    # 5. matrice de confusion de l'ensemble
    predits_ens = (fusion["score_ensemble"].to_numpy() >= 0.5).astype(int)
    tracer_matrice(fusion["vrai"].to_numpy(), predits_ens, "ensemble")


if __name__ == "__main__":
    main()