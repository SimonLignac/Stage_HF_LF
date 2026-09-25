# ===========================================================================
# MIL AVEC ROUTAGE selon le type de prelevement
# ---------------------------------------------------------------------------
# On entraine DEUX modeles INDEPENDANTS, chacun sur TOUTES les lames :
#   - modele_20x   : MIL une seule branche (20x seulement)
#   - modele_concat: MIL multi-echelle (20x + 5x concatenes)
#
# Au moment de la PREDICTION (validation / test), on ROUTE chaque lame :
#   - si BIOPSIE            -> on utilise la prediction du modele_20x
#   - si PIECE CHIRURGICALE -> on utilise la prediction du modele_concat
#
# Idee (Dr Franchet) : sur les biopsies, le 5x n'apporte rien. En entrainant
# un VRAI modele 20x seul (poids non partages avec le 5x), les biopsies
# beneficient d'un modele non "pollue" par l'echelle 5x.
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
NB_EPOCHS = 50
PATIENCE = 18
TAUX_APPRENTISSAGE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 25
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


# --- une branche d'attention (encodeur + attention -> vecteur de lame) ---
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


# --- MODELE 1 : MIL 20x seul ---
class MIL20x(nn.Module):
    def __init__(self, dim_entree, nb_classes):
        super().__init__()
        self.branche = BrancheAttention(dim_entree)
        self.classifieur = nn.Linear(256, nb_classes)
    def forward(self, patches_20x):
        v, poids = self.branche(patches_20x)
        return self.classifieur(v).unsqueeze(0), poids


# --- MODELE 2 : MIL concatenation (20x + 5x) ---
class MILConcat(nn.Module):
    def __init__(self, dim_entree, nb_classes):
        super().__init__()
        self.branche_20x = BrancheAttention(dim_entree)
        self.branche_5x  = BrancheAttention(dim_entree)
        self.classifieur = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, nb_classes),
        )
    def forward(self, patches_20x, patches_5x):
        v20, _ = self.branche_20x(patches_20x)
        v5,  _ = self.branche_5x(patches_5x)
        v_fusion = torch.cat([v20, v5], dim=0)
        return self.classifieur(v_fusion.unsqueeze(0)), None


# --- entraine un modele 20x seul, renvoie le meilleur state ---
def entrainer_20x(lames_train, lames_val):
    modele = MIL20x(DIM_FEATURES, len(CLASSES)).to(DEVICE)
    opt = torch.optim.AdamW(modele.parameters(), lr=TAUX_APPRENTISSAGE, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2, min_lr=1e-6)
    crit = nn.CrossEntropyLoss()
    meilleur = (-1.0, -1.0); meilleur_state = copy.deepcopy(modele.state_dict())
    meilleure_epoch = 0; sans_amelioration = 0

    @torch.no_grad()
    def evaluer(sous_tableau):
        modele.eval(); vr, sc = [], []; tot = 0.0
        for _, r in sous_tableau.iterrows():
            x = torch.from_numpy(charger_lame(r["h5_20x"])).float().to(DEVICE)
            y = torch.tensor([label_vers_indice[r["label"]]]).to(DEVICE)
            sortie, _ = modele(x); tot += crit(sortie, y).item()
            sc.append(torch.softmax(sortie, dim=1)[0, 1].item()); vr.append(label_vers_indice[r["label"]])
        vr = np.array(vr); sc = np.array(sc)
        acc = accuracy_score(vr, (sc >= 0.5).astype(int))
        try: auc = roc_auc_score(vr, sc)
        except ValueError: auc = float("nan")
        return acc, auc, tot/len(sous_tableau)

    print("\n[Modele 20x seul]")
    for epoch in range(1, NB_EPOCHS + 1):
        modele.train(); ordre = generateur.permutation(len(lames_train))
        for i in ordre:
            r = lames_train.iloc[i]
            x = torch.from_numpy(charger_lame(r["h5_20x"])).float().to(DEVICE)
            y = torch.tensor([label_vers_indice[r["label"]]]).to(DEVICE)
            opt.zero_grad(); sortie, _ = modele(x); loss = crit(sortie, y)
            loss.backward(); opt.step()
        val_acc, val_auc, val_loss = evaluer(lames_val)
        sched.step(val_loss)
        aucc = 0.0 if np.isnan(val_auc) else val_auc
        if (val_acc, aucc) > meilleur:
            meilleur = (val_acc, aucc); meilleur_state = copy.deepcopy(modele.state_dict())
            meilleure_epoch = epoch; sans_amelioration = 0
        else: sans_amelioration += 1
        print(f"  epoch {epoch:2d}  val_loss {val_loss:.4f} | val_acc {val_acc:.3f} | val_auc {val_auc:.3f} | patience {sans_amelioration}/{PATIENCE}")
        if sans_amelioration >= PATIENCE:
            print(f"  arret anticipe (meilleure = epoch {meilleure_epoch})"); break
    modele.load_state_dict(meilleur_state)
    return modele


# --- entraine un modele concatenation, renvoie le meilleur state ---
def entrainer_concat(lames_train, lames_val):
    modele = MILConcat(DIM_FEATURES, len(CLASSES)).to(DEVICE)
    opt = torch.optim.AdamW(modele.parameters(), lr=TAUX_APPRENTISSAGE, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2, min_lr=1e-6)
    crit = nn.CrossEntropyLoss()
    meilleur = (-1.0, -1.0); meilleur_state = copy.deepcopy(modele.state_dict())
    meilleure_epoch = 0; sans_amelioration = 0

    @torch.no_grad()
    def evaluer(sous_tableau):
        modele.eval(); vr, sc = [], []; tot = 0.0
        for _, r in sous_tableau.iterrows():
            x20 = torch.from_numpy(charger_lame(r["h5_20x"])).float().to(DEVICE)
            x5  = torch.from_numpy(charger_lame(r["h5_5x"])).float().to(DEVICE)
            y = torch.tensor([label_vers_indice[r["label"]]]).to(DEVICE)
            sortie, _ = modele(x20, x5); tot += crit(sortie, y).item()
            sc.append(torch.softmax(sortie, dim=1)[0, 1].item()); vr.append(label_vers_indice[r["label"]])
        vr = np.array(vr); sc = np.array(sc)
        acc = accuracy_score(vr, (sc >= 0.5).astype(int))
        try: auc = roc_auc_score(vr, sc)
        except ValueError: auc = float("nan")
        return acc, auc, tot/len(sous_tableau)

    print("\n[Modele concatenation 20x + 5x]")
    for epoch in range(1, NB_EPOCHS + 1):
        modele.train(); ordre = generateur.permutation(len(lames_train))
        for i in ordre:
            r = lames_train.iloc[i]
            x20 = torch.from_numpy(charger_lame(r["h5_20x"])).float().to(DEVICE)
            x5  = torch.from_numpy(charger_lame(r["h5_5x"])).float().to(DEVICE)
            y = torch.tensor([label_vers_indice[r["label"]]]).to(DEVICE)
            opt.zero_grad(); sortie, _ = modele(x20, x5); loss = crit(sortie, y)
            loss.backward(); opt.step()
        val_acc, val_auc, val_loss = evaluer(lames_val)
        sched.step(val_loss)
        aucc = 0.0 if np.isnan(val_auc) else val_auc
        if (val_acc, aucc) > meilleur:
            meilleur = (val_acc, aucc); meilleur_state = copy.deepcopy(modele.state_dict())
            meilleure_epoch = epoch; sans_amelioration = 0
        else: sans_amelioration += 1
        print(f"  epoch {epoch:2d}  val_loss {val_loss:.4f} | val_acc {val_acc:.3f} | val_auc {val_auc:.3f} | patience {sans_amelioration}/{PATIENCE}")
        if sans_amelioration >= PATIENCE:
            print(f"  arret anticipe (meilleure = epoch {meilleure_epoch})"); break
    modele.load_state_dict(meilleur_state)
    return modele


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
    print(f"Types : {tableau['type'].value_counts().to_dict()}")
    print(f"Splits : {tableau['split'].value_counts().to_dict()}")

    lames_train = tableau[tableau.split == "train"].reset_index(drop=True)
    lames_val   = tableau[tableau.split == "val"].reset_index(drop=True)
    lames_test  = tableau[tableau.split == "test"].reset_index(drop=True)

    # --- entrainer les DEUX modeles, chacun sur TOUTES les lames ---
    modele_20x    = entrainer_20x(lames_train, lames_val)
    modele_concat = entrainer_concat(lames_train, lames_val)

    # --- PREDICTION AVEC ROUTAGE sur le test ---
    modele_20x.eval(); modele_concat.eval()
    vrais, predits, scores, noms, types, diags = [], [], [], [], [], []
    with torch.no_grad():
        for _, r in lames_test.iterrows():
            est_biopsie = (r["type"] == TYPE_BIOPSIE)
            x20 = torch.from_numpy(charger_lame(r["h5_20x"])).float().to(DEVICE)
            if est_biopsie:
                # BIOPSIE -> modele 20x seul
                sortie, _ = modele_20x(x20)
            else:
                # PIECE -> modele concatenation
                x5 = torch.from_numpy(charger_lame(r["h5_5x"])).float().to(DEVICE)
                sortie, _ = modele_concat(x20, x5)
            proba = torch.softmax(sortie, dim=1)[0].cpu().numpy()
            vrais.append(label_vers_indice[r["label"]])
            predits.append(int(proba.argmax()))
            scores.append(proba[1])
            noms.append(r["slide"]); types.append(r["type"]); diags.append(r["diagnosis"])

    vrais = np.array(vrais); predits = np.array(predits); scores = np.array(scores)

    print("\n--- Lames mal classees (avec routage) ---")
    for i in range(len(vrais)):
        if vrais[i] != predits[i]:
            print(f"  {noms[i]} : vrai={CLASSES[vrais[i]]}, predit={CLASSES[predits[i]]}, "
                  f"score_FL={scores[i]:.3f}, type={types[i]}, diag={diags[i]}")

    acc = accuracy_score(vrais, predits)
    auc = roc_auc_score(vrais, scores)
    print(f"\n===== TEST (ROUTAGE : biopsie->20x, piece->concat) =====")
    print(f"Accuracy : {acc:.3f} | AUC : {auc:.3f}")
    matrice = confusion_matrix(vrais, predits, labels=[0, 1])
    print("          predit_FH  predit_FL")
    print(f"vrai_FH      {matrice[0,0]:5d}    {matrice[0,1]:6d}")
    print(f"vrai_FL      {matrice[1,0]:5d}    {matrice[1,1]:6d}")
    print(classification_report(vrais, predits, target_names=CLASSES, digits=3))

    # matrice image
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(matrice, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASSES); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predit"); ax.set_ylabel("Vrai")
    ax.set_title("Matrice de confusion (TEST, routage)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, matrice[i, j], ha="center", va="center",
                    color="white" if matrice[i, j] > matrice.max()/2 else "black", fontsize=14)
    plt.tight_layout()
    plt.savefig(DOSSIER_SORTIE / "matrice_confusion_routage.png", dpi=130)
    print(f"Matrice -> {DOSSIER_SORTIE / 'matrice_confusion_routage.png'}")

    torch.save(modele_20x.state_dict(),    DOSSIER_SORTIE / "modele_routage_20x.pth")
    torch.save(modele_concat.state_dict(), DOSSIER_SORTIE / "modele_routage_concat.pth")
    print("Modeles sauvegardes (20x + concat).")


if __name__ == "__main__":
    main()