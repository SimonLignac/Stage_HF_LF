# ===========================================================================
# COMPARAISON DE 3 APPROCHES sur le MEME run (meme split, memes modeles)
# + COURBES D'APPRENTISSAGE pour chaque modele
# ---------------------------------------------------------------------------
# 1) H-optimus 20x seul   2) Virchow2 concat   3) Routage mixte
#
# Version CAMELYON : les lames CAMELYON (toutes FH) sont FUSIONNEES avec le
# dataset de base LF (qui contient les FH ET les FL), pour que les splits
# train/val/test contiennent bien les deux classes.
# ===========================================================================

from pathlib import Path
import copy
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- config commune ----
CLASSES = ["FH", "FL"]
MAX_PATCHS_PAR_LAME = 2000
NB_EPOCHS = 200
PATIENCE = 45
TAUX_APPRENTISSAGE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 10034
TYPE_BIOPSIE = "Biopsy"
DOSSIER_SORTIE = Path("/mnt/e/internship_CAMELYON/output_virchow2")            # a verifier

# ---- config dataset de BASE (LF / Oncopole) ----

# ---- config dataset de BASE (LF / Oncopole) ----
BASE_HOPT_MANIFEST_20X = Path("/mnt/e/internship/output_LF/manifest_slides_LF.csv")            # a verifier
BASE_VIR_MANIFEST_20X  = Path("/mnt/e/internship/output_LF_virchow2/manifest_slides_LF.csv")   # a verifier
BASE_VIR_MANIFEST_5X   = Path("/mnt/e/internship/output_LF_virchow2_5x/manifest_slides_5x.csv")# a verifier

# ---- config H-OPTIMUS (CAMELYON) ----
HOPT_MANIFEST_20X = Path("/mnt/e/internship_CAMELYON/output_hoptimus/manifest_slides_LF.csv")
HOPT_ENCODEUR = "h_optimus_1"
HOPT_DIM = 1536

# ---- config VIRCHOW 2 (CAMELYON) ----
VIR_MANIFEST_20X = Path("/mnt/e/internship_CAMELYON/output_virchow2/manifest_slides_CAMELYON.csv")
VIR_MANIFEST_5X  = Path("/mnt/e/internship_CAMELYON/output_virchow2_5x/manifest_slides_5x_CAMELYON.csv")
VIR_ENCODEUR = "virchow_v2"
VIR_DIM = 2560

torch.manual_seed(SEED); np.random.seed(SEED)
label_vers_indice = {"FH": 0, "FL": 1}
generateur = np.random.default_rng(SEED)


def charger_lame(h5_path, encodeur):
    with h5py.File(h5_path, "r") as f:
        feats = f[f"features/{encodeur}"][...]
    if len(feats) > MAX_PATCHS_PAR_LAME:
        idx = generateur.choice(len(feats), MAX_PATCHS_PAR_LAME, replace=False)
        feats = feats[idx]
    return feats


def charger_et_fusionner(chemin_base, chemin_camelyon):
    """Concatene le manifest de base (LF) et le manifest CAMELYON pour un encodeur donne."""
    base = pd.read_csv(chemin_base)
    came = pd.read_csv(chemin_camelyon)
    base["source"] = "LF"
    came["source"] = "CAMELYON"
    fusion = pd.concat([base, came], ignore_index=True)
    # garde-fou : deux lames de sources differentes ne doivent pas avoir le meme nom
    doublons = fusion["slide"][fusion["slide"].duplicated()].unique()
    if len(doublons):
        raise ValueError(
            f"Collision de noms de lames entre les 2 sources : {list(doublons[:5])} "
            f"({len(doublons)} au total). Renomme-les avant de fusionner."
        )
    return fusion.set_index("slide")


class BrancheAttention(nn.Module):
    def __init__(self, dim_entree):
        super().__init__()
        self.encodeur_patch = nn.Sequential(nn.Linear(dim_entree, 256), nn.ReLU(), nn.Dropout(0.3))
        self.attention = nn.Sequential(nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 1))
    def forward(self, patches):
        h = self.encodeur_patch(patches)
        poids = torch.softmax(self.attention(h), dim=0)
        return torch.sum(poids * h, dim=0), poids


class MIL20x(nn.Module):
    def __init__(self, dim_entree, nb_classes):
        super().__init__()
        self.branche = BrancheAttention(dim_entree)
        self.classifieur = nn.Linear(256, nb_classes)
    def forward(self, patches_20x):
        v, poids = self.branche(patches_20x)
        return self.classifieur(v).unsqueeze(0), poids


class MILConcat(nn.Module):
    def __init__(self, dim_entree, nb_classes):
        super().__init__()
        self.branche_20x = BrancheAttention(dim_entree)
        self.branche_5x  = BrancheAttention(dim_entree)
        self.classifieur = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3),
                                         nn.Linear(128, nb_classes))
    def forward(self, patches_20x, patches_5x):
        v20, _ = self.branche_20x(patches_20x)
        v5,  _ = self.branche_5x(patches_5x)
        return self.classifieur(torch.cat([v20, v5], dim=0).unsqueeze(0)), None


# --- trace les courbes d'apprentissage d'un modele ---
def tracer_courbes(historique, nom, fichier, meilleure_epoch):
    nb = len(historique["train_loss"])
    ep = range(1, nb + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    # loss train vs val
    axes[0].plot(ep, historique["train_loss"], marker="o", ms=3, label="train")
    axes[0].plot(ep, historique["val_loss"], marker="o", ms=3, color="orange", label="validation")
    axes[0].axvline(meilleure_epoch, color="grey", ls="--", alpha=.6)
    axes[0].set_title(f"Loss train vs validation ({nom})")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].grid(alpha=.3); axes[0].legend()
    # accuracy
    axes[1].plot(ep, historique["val_acc"], color="green", marker="o", ms=3)
    axes[1].axvline(meilleure_epoch, color="grey", ls="--", alpha=.6)
    axes[1].set_title(f"Accuracy validation ({nom})")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy"); axes[1].grid(alpha=.3)
    # auc
    axes[2].plot(ep, historique["val_auc"], color="purple", marker="o", ms=3)
    axes[2].axvline(meilleure_epoch, color="grey", ls="--", alpha=.6)
    axes[2].set_title(f"AUC validation ({nom})")
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("AUC"); axes[2].grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(DOSSIER_SORTIE / fichier, dpi=130); plt.close()
    print(f"Courbes -> {DOSSIER_SORTIE / fichier}")


def entrainer_hopt_20x(train, val):
    modele = MIL20x(HOPT_DIM, len(CLASSES)).to(DEVICE)
    opt = torch.optim.AdamW(modele.parameters(), lr=TAUX_APPRENTISSAGE, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2, min_lr=1e-6)
    crit = nn.CrossEntropyLoss()
    meilleur = (-1., -1.); best_state = copy.deepcopy(modele.state_dict()); best_ep = 0; sans = 0
    historique = {"train_loss": [], "val_loss": [], "val_acc": [], "val_auc": []}

    @torch.no_grad()
    def evaluer(sub):
        modele.eval(); vr, sc = [], []; tot = 0.
        for _, r in sub.iterrows():
            x = torch.from_numpy(charger_lame(r["h5_hopt"], HOPT_ENCODEUR)).float().to(DEVICE)
            y = torch.tensor([label_vers_indice[r["label"]]]).to(DEVICE)
            s, _ = modele(x); tot += crit(s, y).item()
            sc.append(torch.softmax(s, 1)[0, 1].item()); vr.append(label_vers_indice[r["label"]])
        vr, sc = np.array(vr), np.array(sc)
        acc = accuracy_score(vr, (sc >= .5).astype(int))
        try: auc = roc_auc_score(vr, sc)
        except ValueError: auc = float("nan")
        return acc, auc, tot/len(sub)

    print("\n[Entrainement H-optimus 20x seul]")
    for ep in range(1, NB_EPOCHS + 1):
        modele.train(); total_train = 0.
        for i in generateur.permutation(len(train)):
            r = train.iloc[i]
            x = torch.from_numpy(charger_lame(r["h5_hopt"], HOPT_ENCODEUR)).float().to(DEVICE)
            y = torch.tensor([label_vers_indice[r["label"]]]).to(DEVICE)
            opt.zero_grad(); s, _ = modele(x); loss = crit(s, y); loss.backward(); opt.step()
            total_train += loss.item()
        train_loss = total_train / len(train)
        va, au, vl = evaluer(val); sched.step(vl)
        historique["train_loss"].append(train_loss); historique["val_loss"].append(vl)
        historique["val_acc"].append(va); historique["val_auc"].append(au)
        auc_c = 0. if np.isnan(au) else au
        if (va, auc_c) > meilleur:
            meilleur = (va, auc_c); best_state = copy.deepcopy(modele.state_dict()); best_ep = ep; sans = 0
        else: sans += 1
        print(f"  epoch {ep:2d}  train_loss {train_loss:.4f} | val_loss {vl:.4f} | val_acc {va:.3f} | val_auc {au:.3f} | patience {sans}/{PATIENCE}")
        if sans >= PATIENCE:
            print(f"  arret (meilleure = epoch {best_ep})"); break
    modele.load_state_dict(best_state)
    tracer_courbes(historique, "H-optimus 20x", "courbes_hopt20x.png", best_ep)
    torch.save(modele.state_dict(), DOSSIER_SORTIE / "modele_hopt20x.pth")
    print(f"Modele sauvegarde -> {DOSSIER_SORTIE / 'modele_hopt20x.pth'}")
    return modele


def entrainer_vir_concat(train, val):
    modele = MILConcat(VIR_DIM, len(CLASSES)).to(DEVICE)
    opt = torch.optim.AdamW(modele.parameters(), lr=TAUX_APPRENTISSAGE, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2, min_lr=1e-6)
    crit = nn.CrossEntropyLoss()
    meilleur = (-1., -1.); best_state = copy.deepcopy(modele.state_dict()); best_ep = 0; sans = 0
    historique = {"train_loss": [], "val_loss": [], "val_acc": [], "val_auc": []}

    @torch.no_grad()
    def evaluer(sub):
        modele.eval(); vr, sc = [], []; tot = 0.
        for _, r in sub.iterrows():
            x20 = torch.from_numpy(charger_lame(r["h5_vir20"], VIR_ENCODEUR)).float().to(DEVICE)
            x5  = torch.from_numpy(charger_lame(r["h5_vir5"],  VIR_ENCODEUR)).float().to(DEVICE)
            y = torch.tensor([label_vers_indice[r["label"]]]).to(DEVICE)
            s, _ = modele(x20, x5); tot += crit(s, y).item()
            sc.append(torch.softmax(s, 1)[0, 1].item()); vr.append(label_vers_indice[r["label"]])
        vr, sc = np.array(vr), np.array(sc)
        acc = accuracy_score(vr, (sc >= .5).astype(int))
        try: auc = roc_auc_score(vr, sc)
        except ValueError: auc = float("nan")
        return acc, auc, tot/len(sub)

    print("\n[Entrainement Virchow2 concatenation 20x + 5x]")
    for ep in range(1, NB_EPOCHS + 1):
        modele.train(); total_train = 0.
        for i in generateur.permutation(len(train)):
            r = train.iloc[i]
            x20 = torch.from_numpy(charger_lame(r["h5_vir20"], VIR_ENCODEUR)).float().to(DEVICE)
            x5  = torch.from_numpy(charger_lame(r["h5_vir5"],  VIR_ENCODEUR)).float().to(DEVICE)
            y = torch.tensor([label_vers_indice[r["label"]]]).to(DEVICE)
            opt.zero_grad(); s, _ = modele(x20, x5); loss = crit(s, y); loss.backward(); opt.step()
            total_train += loss.item()
        train_loss = total_train / len(train)
        va, au, vl = evaluer(val); sched.step(vl)
        historique["train_loss"].append(train_loss); historique["val_loss"].append(vl)
        historique["val_acc"].append(va); historique["val_auc"].append(au)
        auc_c = 0. if np.isnan(au) else au
        if (va, auc_c) > meilleur:
            meilleur = (va, auc_c); best_state = copy.deepcopy(modele.state_dict()); best_ep = ep; sans = 0
        else: sans += 1
        print(f"  epoch {ep:2d}  train_loss {train_loss:.4f} | val_loss {vl:.4f} | val_acc {va:.3f} | val_auc {au:.3f} | patience {sans}/{PATIENCE}")
        if sans >= PATIENCE:
            print(f"  arret (meilleure = epoch {best_ep})"); break
    modele.load_state_dict(best_state)
    tracer_courbes(historique, "Virchow2 concat", "courbes_vir_concat.png", best_ep)
    torch.save(modele.state_dict(), DOSSIER_SORTIE / "modele_vir_concat.pth")
    print(f"Modele sauvegarde -> {DOSSIER_SORTIE / 'modele_vir_concat.pth'}")
    return modele


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
        auc = float("nan")
    print(f"\n===== {nom} =====")
    print(f"Accuracy : {acc:.3f} | AUC : {auc:.3f}")
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
    # --- Fusion des manifests : dataset de base (FH+FL) + CAMELYON (FH) ---
    hopt = charger_et_fusionner(BASE_HOPT_MANIFEST_20X, HOPT_MANIFEST_20X)
    v20  = charger_et_fusionner(BASE_VIR_MANIFEST_20X,  VIR_MANIFEST_20X)
    v5   = charger_et_fusionner(BASE_VIR_MANIFEST_5X,   VIR_MANIFEST_5X)

    communes = hopt.index.intersection(v20.index).intersection(v5.index)
    print(f"{len(communes)} lames communes aux 3 manifests")

    incoherent = 0; lignes = []
    for slide in communes:
        if hopt.loc[slide, "split"] != v20.loc[slide, "split"]:
            incoherent += 1
        lignes.append({
            "slide": slide, "label": v20.loc[slide, "label"], "split": v20.loc[slide, "split"],
            "type": v20.loc[slide, "type"],
            "diagnosis": v20.loc[slide, "diagnosis"] if "diagnosis" in v20.columns else "",
            "source": v20.loc[slide, "source"] if "source" in v20.columns else "",
            "h5_hopt": hopt.loc[slide, "h5_path"], "h5_vir20": v20.loc[slide, "h5_path"],
            "h5_vir5": v5.loc[slide, "h5_path"],
        })
    if incoherent > 0:
        print(f"[!] {incoherent} lames avec split different -> routage invalide, arret."); return
    print("OK : meme split entre h_optimus et virchow")

    tableau = pd.DataFrame(lignes)
    print(f"Sources : {tableau['source'].value_counts().to_dict()}")
    print(f"Types : {tableau['type'].value_counts().to_dict()}")
    print(f"Splits : {tableau['split'].value_counts().to_dict()}")
    # comptes label x split : LE controle a regarder pour verifier que FH ET FL
    # sont bien presents dans val et test.
    print("Repartition label par split :")
    print(tableau.groupby("split")["label"].value_counts())

    train = tableau[tableau.split == "train"].reset_index(drop=True)
    val   = tableau[tableau.split == "val"].reset_index(drop=True)
    test  = tableau[tableau.split == "test"].reset_index(drop=True)

    modele_hopt = entrainer_hopt_20x(train, val)
    modele_vir  = entrainer_vir_concat(train, val)
    modele_hopt.eval(); modele_vir.eval()

    v1 = ([], [], [], [], [], [])
    with torch.no_grad():
        for _, r in test.iterrows():
            x = torch.from_numpy(charger_lame(r["h5_hopt"], HOPT_ENCODEUR)).float().to(DEVICE)
            p = torch.softmax(modele_hopt(x)[0], 1)[0].cpu().numpy()
            v1[0].append(label_vers_indice[r["label"]]); v1[1].append(int(p.argmax())); v1[2].append(p[1])
            v1[3].append(r["slide"]); v1[4].append(r["type"]); v1[5].append(r["diagnosis"])
    a1, au1 = rapport("H-optimus 20x seul", "matrice_hopt20x.png", *v1)

    v2 = ([], [], [], [], [], [])
    with torch.no_grad():
        for _, r in test.iterrows():
            x20 = torch.from_numpy(charger_lame(r["h5_vir20"], VIR_ENCODEUR)).float().to(DEVICE)
            x5  = torch.from_numpy(charger_lame(r["h5_vir5"],  VIR_ENCODEUR)).float().to(DEVICE)
            p = torch.softmax(modele_vir(x20, x5)[0], 1)[0].cpu().numpy()
            v2[0].append(label_vers_indice[r["label"]]); v2[1].append(int(p.argmax())); v2[2].append(p[1])
            v2[3].append(r["slide"]); v2[4].append(r["type"]); v2[5].append(r["diagnosis"])
    a2, au2 = rapport("Virchow2 concatenation", "matrice_vir_concat.png", *v2)

    v3 = ([], [], [], [], [], [])
    with torch.no_grad():
        for _, r in test.iterrows():
            if r["type"] == TYPE_BIOPSIE:
                x = torch.from_numpy(charger_lame(r["h5_hopt"], HOPT_ENCODEUR)).float().to(DEVICE)
                p = torch.softmax(modele_hopt(x)[0], 1)[0].cpu().numpy()
            else:
                x20 = torch.from_numpy(charger_lame(r["h5_vir20"], VIR_ENCODEUR)).float().to(DEVICE)
                x5  = torch.from_numpy(charger_lame(r["h5_vir5"],  VIR_ENCODEUR)).float().to(DEVICE)
                p = torch.softmax(modele_vir(x20, x5)[0], 1)[0].cpu().numpy()
            v3[0].append(label_vers_indice[r["label"]]); v3[1].append(int(p.argmax())); v3[2].append(p[1])
            v3[3].append(r["slide"]); v3[4].append(r["type"]); v3[5].append(r["diagnosis"])
    a3, au3 = rapport("Double encodeur (biopsie->Hopt, piece->Vir concat)", "matrice_double_encodeur.png", *v3)

    print("\n" + "="*60)
    print("RESUME COMPARATIF (meme test, meme run)")
    print("="*60)
    print(f"  1) H-optimus 20x seul       : acc {a1:.3f} | AUC {au1:.3f}")
    print(f"  2) Virchow2 concatenation   : acc {a2:.3f} | AUC {au2:.3f}")
    print(f"  3) Double encodeur           : acc {a3:.3f} | AUC {au3:.3f}")


if __name__ == "__main__":
    main()