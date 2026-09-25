# ===========================================================================
# ENSEMBLE MIL : 20x + 5x (multi-echelle avec attention)
# ---------------------------------------------------------------------------
# On entraine DEUX modeles MIL a attention (un sur 20x, un sur 5x),
# on recupere le score de chaque lame de test pour chacun, et on MOYENNE.
# But : voir si combiner les echelles avec le MIL fait mieux que chacune seule.
# ===========================================================================

from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score

# --- REGLAGES COMMUNS ---
DIM_FEATURES = 1536
CLASSES = ["FH", "FL"]
MAX_PATCHS_PAR_LAME = 2000    # en MIL on peut en garder plus (l'attention fait le tri)
NB_EPOCHS = 12               # MIL : plus d'epochs car peu de lames (355)
TAUX_APPRENTISSAGE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# Les deux manifests (meme split, echelles differentes)
MANIFEST_20X = Path("/mnt/d/internship/output/manifest_slides.csv")
MANIFEST_5X  = Path("/mnt/e/internship/output_5x/manifest_slides_5x.csv")
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


# --- le modele MIL avec attention (identique a ton script MIL) ---
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


# --- entraine un MIL sur un manifest, renvoie les SCORES par lame du test ---
def entrainer_et_scorer_MIL(manifest_path, nom):
    print(f"\n===== Modele MIL {nom} ({manifest_path.name}) =====")
    tableau = pd.read_csv(manifest_path)
    lames_train = tableau[tableau.split == "train"].reset_index(drop=True)

    modele = MILAttention(DIM_FEATURES, len(CLASSES)).to(DEVICE)
    opt = torch.optim.AdamW(modele.parameters(), lr=TAUX_APPRENTISSAGE, weight_decay=1e-2)
    crit = nn.CrossEntropyLoss()

    # entrainement lame par lame
    for epoch in range(1, NB_EPOCHS + 1):
        modele.train()
        ordre = generateur.permutation(len(lames_train))
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
        print(f"  epoch {epoch}/{NB_EPOCHS} fait")

    # scores par lame sur le TEST
    modele.eval()
    lames_test = tableau[tableau.split == "test"].reset_index(drop=True)
    scores, vrais, noms = [], [], []
    with torch.no_grad():
        for _, r in lames_test.iterrows():
            feats = charger_lame(r["h5_path"])
            x = torch.from_numpy(feats).float().to(DEVICE)
            sortie, _ = modele(x)
            proba = torch.softmax(sortie, dim=1)[0].cpu().numpy()
            scores.append(proba[1])                 # proba FL de la lame
            vrais.append(label_vers_indice[r["label"]])
            noms.append(r["slide"])
    return pd.DataFrame({"slide": noms, "vrai": vrais, f"score_{nom}": scores})


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
    # 1. entraine les deux MIL et recupere leurs scores test
    df20 = entrainer_et_scorer_MIL(MANIFEST_20X, "20x")
    df5  = entrainer_et_scorer_MIL(MANIFEST_5X,  "5x")

    # 2. on apparie par nom de lame (meme test, meme split)
    fusion = df20.merge(df5[["slide", "score_5x"]], on="slide")
    print(f"\n{len(fusion)} lames de test appariees entre 20x et 5x")

    # 3. l'ensemble = moyenne des deux scores
    fusion["score_ensemble"] = (fusion["score_20x"] + fusion["score_5x"]) / 2

    # 4. on evalue chacun + l'ensemble
    print("\n" + "="*60)
    print("COMPARAISON FINALE MIL (sur le meme test)")
    print("="*60)
    evaluer_scores(fusion["vrai"], fusion["score_20x"],       "MIL 20x seul")
    evaluer_scores(fusion["vrai"], fusion["score_5x"],        "MIL 5x seul")
    evaluer_scores(fusion["vrai"], fusion["score_ensemble"],  "ENSEMBLE MIL 20x+5x (moyenne)")


if __name__ == "__main__":
    main()