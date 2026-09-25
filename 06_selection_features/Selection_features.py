# ===========================================================================
# SELECTION DE FEATURES (niveau lame) avec scikit-learn
# ---------------------------------------------------------------------------
# Objectif  : parmi les 1536 dimensions de h_optimus,
# identifier lesquelles sont utiles pour distinguer FH de FL.
#
# Demarche :
#   1) chaque lame -> un vecteur (MOYENNE de ses patches) : format tabulaire
#   2) on standardise (moyenne 0, variance 1) sur le TRAIN uniquement
#   3) deux methodes de selection :
#        a) SelectKBest (test F univarie) : features les plus correlees au label
#        b) Regression logistique L1 (Lasso) : features gardees par le modele
#   4) on VERIFIE si un petit sous-ensemble suffit : on compare l'AUC (validation
#      croisee) avec toutes les features vs avec les K meilleures.
#
# IMPORTANT : selection sur le TRAIN uniquement, refaite dans chaque pli de la CV (Pipeline).
# ===========================================================================

from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- config ---
MANIFEST = Path("/mnt/e/internship/output_LF/manifest_slides_LF.csv")
ENCODEUR = "h_optimus_1"
DIM_FEATURES = 1536
DOSSIER_SORTIE = Path("/mnt/e/internship/output_LF")
label_vers_indice = {"FH": 0, "FL": 1}
SEED = 42


def vecteur_lame_moyen(h5_path):
    """Represente une lame par la MOYENNE de ses patches (vecteur de DIM_FEATURES)."""
    with h5py.File(h5_path, "r") as f:
        feats = f[f"features/{ENCODEUR}"][...]     # (N_patches, DIM_FEATURES)
    return feats.mean(axis=0)                        # (DIM_FEATURES,)


def main():
    tableau = pd.read_csv(MANIFEST)
    print(f"{len(tableau)} lames dans le manifest")

    # 1. construire la matrice X (lames x features) et le vecteur y (labels)
    print("Agregation des lames (moyenne des patches)...")
    X, y, groupes = [], [], []
    for _, r in tableau.iterrows():
        X.append(vecteur_lame_moyen(r["h5_path"]))
        y.append(label_vers_indice[r["label"]])
        groupes.append(r["case_id"])       # pour la validation croisee par patient
    X = np.array(X); y = np.array(y); groupes = np.array(groupes)
    print(f"Matrice X : {X.shape} (lames x features) | {y.sum()} FL, {len(y)-y.sum()} FH")

    # on travaille sur le TRAIN pour la selection (pas de fuite du test)
    est_train = (tableau["split"] == "train").to_numpy()
    Xtr, ytr, gtr = X[est_train], y[est_train], groupes[est_train]
    print(f"Train : {Xtr.shape[0]} lames")

    # 2. standardiser (ajuste sur le train)
    scaler = StandardScaler().fit(Xtr)
    Xtr_std = scaler.transform(Xtr)

    # 3a. SelectKBest : features les plus correlees au label (test F)
    print("\n=== Methode A : SelectKBest (test F univarie) ===")
    selecteur = SelectKBest(score_func=f_classif, k=50).fit(Xtr_std, ytr)
    scores = selecteur.scores_                       # score par feature
    top_kbest = np.argsort(scores)[::-1][:20]        # les 20 meilleures
    print("Top 20 features (SelectKBest) : indices et scores F :")
    for rang, idx in enumerate(top_kbest, 1):
        print(f"  {rang:2d}. feature #{idx:4d}  (score F = {scores[idx]:.1f})")

    # 3b. Regression logistique L1 : features gardees par le modele
    print("\n=== Methode B : Regression logistique L1 (Lasso) ===")
    logreg = LogisticRegression(penalty="l1", solver="liblinear", C=0.1, random_state=SEED)
    logreg.fit(Xtr_std, ytr)
    coefs = np.abs(logreg.coef_[0])                  # importance = |coefficient|
    nb_gardees = (coefs > 0).sum()
    top_l1 = np.argsort(coefs)[::-1][:20]
    print(f"Features gardees par L1 (coef non nul) : {nb_gardees} / {DIM_FEATURES}")
    print("Top 20 features (L1) : indices et |coef| :")
    for rang, idx in enumerate(top_l1, 1):
        print(f"  {rang:2d}. feature #{idx:4d}  (|coef| = {coefs[idx]:.3f})")

    # 4. un petit sous-ensemble suffit-il ? On compare l'AUC (validation croisee
    #    par patient) avec differents nombres de features.
    print("\n=== Un sous-ensemble de features suffit-il ? (AUC en validation croisee) ===")
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    valeurs_k = [10, 25, 50, 100, 200, 500, DIM_FEATURES]
    aucs_moy = []
    for k in valeurs_k:
        # Pipeline : standardisation et selection sont REFAITES dans chaque pli,
        # sur les 4 plis d'entrainement uniquement (le pli d'evaluation ne sert
        # jamais a choisir les features -> pas de fuite)
        etapes = [StandardScaler()]
        if k < DIM_FEATURES:
            etapes.append(SelectKBest(score_func=f_classif, k=k))
        etapes.append(LogisticRegression(max_iter=1000, random_state=SEED))
        modele = make_pipeline(*etapes)
        scores_cv = cross_val_score(modele, Xtr, ytr, groups=gtr, cv=cv, scoring="roc_auc")
        aucs_moy.append(scores_cv.mean())
        print(f"  k = {k:4d} features -> AUC moyenne (CV) = {scores_cv.mean():.3f} (+/- {scores_cv.std():.3f})")

    # trace : AUC en fonction du nombre de features
    plt.figure(figsize=(8, 5))
    plt.plot(valeurs_k, aucs_moy, marker="o")
    plt.xscale("log")
    plt.xlabel("Nombre de features gardees (echelle log)")
    plt.ylabel("AUC moyenne (validation croisee)")
    plt.title("Performance selon le nombre de features (h_optimus, niveau lame)")
    plt.grid(alpha=.3)
    plt.tight_layout()
    chemin = DOSSIER_SORTIE / "selection_features_auc.png"
    plt.savefig(chemin, dpi=130)
    print(f"\nCourbe AUC vs nb features -> {chemin}")
    plt.close()

    # meme trace, mais avec l'axe AUC fixe entre 0 et 1
    plt.figure(figsize=(8, 5))
    plt.plot(valeurs_k, aucs_moy, marker="o")
    plt.xscale("log")
    plt.ylim(0, 1)
    plt.xlabel("Nombre de features gardees (echelle log)")
    plt.ylabel("AUC moyenne (validation croisee)")
    plt.title("Performance selon le nombre de features (AUC sur [0, 1])")
    plt.grid(alpha=.3)
    plt.tight_layout()
    chemin_01 = DOSSIER_SORTIE / "selection_features_auc_0_1.png"
    plt.savefig(chemin_01, dpi=130)
    plt.close()
    print(f"Courbe AUC vs nb features (echelle 0-1) -> {chemin_01}")

    # sauvegarde des indices des meilleures features (pour reutilisation eventuelle)
    np.save(DOSSIER_SORTIE / "top_features_kbest.npy", np.argsort(scores)[::-1][:100])
    print(f"Indices des 100 meilleures features (KBest) sauvegardes.")


if __name__ == "__main__":
    main()