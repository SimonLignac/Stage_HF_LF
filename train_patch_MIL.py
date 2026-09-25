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
matplotlib.use("Agg")            # mode "sans fenetre" : on sauvegarde en image au lieu d'afficher
import matplotlib.pyplot as plt


# config

DOSSIER_SORTIE = Path("/mnt/e/internship/output_LF")
MANIFEST = DOSSIER_SORTIE / "manifest_slides_LF.csv"
ENCODEUR = "h_optimus_1"
DIM_FEATURES = 1536
CLASSES = ["FH", "FL"]
MAX_PATCHS_PAR_LAME = 2000    # en MIL on peut en garder plus : l'attention fera le tri
NB_EPOCHS = 14              # plafond : l'early stopping coupera avant si besoin
PATIENCE = 14                # nb d'epochs sans amelioration avant d'arreter
TAUX_APPRENTISSAGE = 1e-4    #  vitesse d'apprentissage
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"   # GPU si dispo, sinon CPU
SEED = 82                 # seed du hasard (pour la reproductibilite)

torch.manual_seed(SEED); np.random.seed(SEED)      # applique la seed
label_vers_indice = {"FH": 0, "FL": 1}
generateur = np.random.default_rng(SEED)             # generateur de hasard pour l'echantillonnage


# FONCTION : charger UNE lame entiere (tous ses patches d'un coup)
# En MIL, on ne melange plus les patches : une lame = un exemple = un "sac" de patches

def charger_lame(h5_path):
    with h5py.File(h5_path, "r") as f:               # ouvre le fichier H5 de la lame ("r" = read)
        feats = f[f"features/{ENCODEUR}"][...]       # lit tous ses vecteurs de features (le [...] charge tout)
    if len(feats) > MAX_PATCHS_PAR_LAME:             # si trop de patches...
        idx = generateur.choice(len(feats), MAX_PATCHS_PAR_LAME, replace=False)  # tire au hasard
        feats = feats[idx]                            # garde ceux-la
    return feats                                     # tableau (nb_patches, 1536) pour UNE lame


# CLASSE : le modele MIL avec attention
# Difference avec le MLP : au lieu de classer chaque patch, on donne un poids
# d'importance a chaque patch (attention), puis on agrege par moyenne ponderee.

class MILAttention(nn.Module):
    def __init__(self, dim_entree, nb_classes):
        super().__init__()                            # initialisation PyTorch obligatoire
        # 1. Encodeur leger : transforme chaque patch (1536 -> 256)
        self.encodeur_patch = nn.Sequential(
            nn.Linear(dim_entree, 256),   # transforme 1536 -> 256
            nn.ReLU(),                    # non-linearite
            nn.Dropout(0.3),              # anti-surapprentissage
        )
        # 2. Module d'ATTENTION : donne un score d'importance a chaque patch
        self.attention = nn.Sequential(
            nn.Linear(256, 128),   # transforme 256 -> 128
            nn.Tanh(),             # non-linearite classique pour l'attention
            nn.Linear(128, 1),     # un seul score d'importance par patch
        )
        # 3. Classifieur final : decide FH/FL a partir du vecteur agrege de la lame
        self.classifieur = nn.Linear(256, nb_classes)   # transforme 256 -> 2 scores (FH, FL)

    def forward(self, patches):                       # patches : (nb_patches, 1536) pour UNE lame
        h = self.encodeur_patch(patches)              # (nb_patches, 256) : chaque patch encode
        scores_attention = self.attention(h)          # (nb_patches, 1) : un score par patch
        poids = torch.softmax(scores_attention, dim=0)  # (nb_patches, 1) : poids normalises (somme = 1)
        vecteur_lame = torch.sum(poids * h, dim=0)    # (256,) : moyenne ponderee des patches
        sortie = self.classifieur(vecteur_lame)       # (2,) : scores FH/FL de la lame
        return sortie.unsqueeze(0), poids             # on renvoie aussi les poids (pour l'interpretabilite)


# FONCTION PRINCIPALE

def main():
    tableau = pd.read_csv(MANIFEST)                   # lit le manifest
    print(f"{len(tableau)} lames | splits : {tableau['split'].value_counts().to_dict()}")

    # En MIL on garde les tableaux de lames (pas de DataLoader de patches)
    lames_train = tableau[tableau.split == "train"].reset_index(drop=True)
    lames_val   = tableau[tableau.split == "val"].reset_index(drop=True)

    modele = MILAttention(DIM_FEATURES, len(CLASSES)).to(DEVICE)  # cree le modele (dim_entree, nb_classes)
    optimiseur = torch.optim.AdamW(modele.parameters(), lr=TAUX_APPRENTISSAGE, weight_decay=1e-2)  # optimiseur
    critere = nn.CrossEntropyLoss()                          # fonction de perte (mesure l'erreur)

    # scheduler : divise le LR par 2 quand la val_loss stagne 2 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiseur, mode="min", factor=0.5, patience=2, min_lr=1e-6
    )

    # pour retenir la meilleure epoch (accuracy d'abord, AUC pour departager)
    meilleur_score = (-1.0, -1.0)
    meilleur_state = copy.deepcopy(modele.state_dict())
    meilleure_epoch = 0
    sans_amelioration = 0        # compteur pour l'early stopping

    # Dictionnaire pour garder l'historique (1 valeur par epoch)
    # on ajoute "val_loss" pour suivre le surapprentissage
    historique = {"train_loss": [], "val_loss": [], "val_acc": [], "val_auc": []}

    # Fonction interne : evalue un groupe de lames (niveau LAME, avec l'attention)
    # Elle calcule aussi la LOSS moyenne du groupe (pour la val_loss)
    @torch.no_grad()                                  # pas de gradients : on ne fait que mesurer
    def metriques_lame(sous_tableau, afficher_erreurs=False):
        modele.eval()                                 # mode evaluation (Dropout desactive)
        vrais, predits, scores, noms = [], [], [], []
        total_loss = 0.0                              # pour accumuler la loss du groupe
        for _, r in sous_tableau.iterrows():          # pour chaque lame
            feats = charger_lame(r["h5_path"])        # charge tous ses patches
            noms.append(Path(r["h5_path"]).stem)      # garde le nom de la lame (sans chemin ni extension)
            x = torch.from_numpy(feats).float().to(DEVICE)
            y = torch.tensor([label_vers_indice[r["label"]]]).to(DEVICE)  # label de la lame
            sortie, _ = modele(x)                     # (1, 2) : le modele decide pour la lame entiere
            total_loss += critere(sortie, y).item()   # loss de cette lame (meme critere que l'entrainement)
            proba = torch.softmax(sortie, dim=1)[0].cpu().numpy()  # [proba_FH, proba_FL]
            vrais.append(label_vers_indice[r["label"]])   # vrai label
            predits.append(int(proba.argmax()))       # label predit
            scores.append(proba[1])                    # proba FL (pour l'AUC)
        vrais = np.array(vrais); predits = np.array(predits)
        if afficher_erreurs:
            print("\n--- Lames mal classees ---")
            for i in range(len(vrais)):
                if vrais[i] != predits[i]:    # erreur
                    vrai_label = CLASSES[vrais[i]]
                    pred_label = CLASSES[predits[i]]
                    print(f"  {noms[i]} : vrai={vrai_label}, predit={pred_label}, "
                                f"score_FL={scores[i]:.3f}")
        acc = accuracy_score(vrais, predits)          # calcule l'accuracy
        try:    auc = roc_auc_score(vrais, scores)     # calcule l'AUC
        except ValueError: auc = float("nan")          # si une seule classe, AUC impossible
        loss = total_loss / len(sous_tableau)         # loss moyenne sur les lames du groupe
        return acc, auc, loss, vrais, predits, scores

    # LA BOUCLE D'ENTRAINEMENT (lame par lame, pas paquet par paquet)
    print("\nEntrainement MIL :")
    for epoch in range(1, NB_EPOCHS + 1):             # repete NB_EPOCHS fois
        modele.train(); total = 0.0                    # mode entrainement (Dropout actif)
        ordre = generateur.permutation(len(lames_train))  # melange l'ordre des lames a chaque epoch
        for i in ordre:                                # parcourt les lames une par une
            r = lames_train.iloc[i]
            feats = charger_lame(r["h5_path"])        # charge tous les patches de la lame
            x = torch.from_numpy(feats).float().to(DEVICE)
            y = torch.tensor([label_vers_indice[r["label"]]]).to(DEVICE)  # le label de la lame

            optimiseur.zero_grad()                     # nettoie les gradients
            sortie, _ = modele(x)                      # (1, 2) : prediction pour la lame
            loss = critere(sortie, y)                  # calcule l'erreur
            loss.backward()                            # calcule comment corriger les poids
            optimiseur.step()                          # applique la correction (apprentissage)
            total += loss.item()                       # accumule la perte
        train_loss = total / len(lames_train)          # perte moyenne d'entrainement

        # mesures sur la validation a la fin de l'epoch (dont la val_loss)
        val_acc, val_auc, val_loss, *_ = metriques_lame(lames_val)
        historique["train_loss"].append(train_loss)
        historique["val_loss"].append(val_loss)
        historique["val_acc"].append(val_acc)
        historique["val_auc"].append(val_auc)

        # learning rate adaptatif : on suit la val_loss
        scheduler.step(val_loss)
        lr_actuel = optimiseur.param_groups[0]["lr"]

        # meilleure epoch : accuracy en priorite, AUC en cas d'egalite
        auc_comparable = 0.0 if np.isnan(val_auc) else val_auc
        if (val_acc, auc_comparable) > meilleur_score:
            meilleur_score = (val_acc, auc_comparable)
            meilleur_state = copy.deepcopy(modele.state_dict())
            meilleure_epoch = epoch
            sans_amelioration = 0                     # on a progresse : compteur remis a zero
        else:
            sans_amelioration += 1                    # pas d'amelioration cette epoch

        print(f"  epoch {epoch:2d}/{NB_EPOCHS}  train_loss {train_loss:.4f} | "
              f"val_loss {val_loss:.4f} | val_acc {val_acc:.3f} | "
              f"val_auc {val_auc:.3f} | lr {lr_actuel:.1e} | "
              f"patience {sans_amelioration}/{PATIENCE}")

        # early stopping : on coupe si la validation ne progresse plus depuis PATIENCE epochs
        if sans_amelioration >= PATIENCE:
            print(f"  arret anticipe : pas d'amelioration depuis {PATIENCE} epochs "
                  f"(meilleure = epoch {meilleure_epoch})")
            break

    # on recharge les poids de la meilleure epoch avant d'evaluer sur le test
    modele.load_state_dict(meilleur_state)
    print(f"\nMeilleure epoch retenue : {meilleure_epoch} "
          f"(val_acc {meilleur_score[0]:.3f}, val_auc {meilleur_score[1]:.3f})")

    # --- TRACE DES COURBES ---
    # nb d'epochs REELLEMENT effectuees (peut etre < NB_EPOCHS a cause de l'early stopping)
    nb_faites = len(historique["train_loss"])
    liste_epochs = range(1, nb_faites + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))   # 3 graphiques cote a cote

    # graphique 1 : loss train ET validation (pour voir le surapprentissage)
    axes[0].plot(liste_epochs, historique["train_loss"], marker="o", ms=3, label="train")
    axes[0].plot(liste_epochs, historique["val_loss"], marker="o", ms=3, color="orange", label="validation")
    axes[0].axvline(meilleure_epoch, color="grey", ls="--", alpha=.6)   # meilleure epoch
    axes[0].set_title("Loss train vs validation (MIL)")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].grid(alpha=.3)
    axes[0].legend()

    axes[1].plot(liste_epochs, historique["val_acc"], color="green", marker="o", ms=3)
    axes[1].set_title("Accuracy validation (MIL)")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy"); axes[1].grid(alpha=.3)

    axes[2].plot(liste_epochs, historique["val_auc"], color="purple", marker="o", ms=3)
    axes[2].set_title("AUC validation (MIL)")
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("AUC"); axes[2].grid(alpha=.3)

    plt.tight_layout()
    chemin_courbes = DOSSIER_SORTIE / "courbes_MIL.png"
    plt.savefig(chemin_courbes, dpi=130)
    print(f"\nCourbes enregistrees -> {chemin_courbes}")

    # --- EVALUATION FINALE SUR LE TEST ---
    lames_test = tableau[tableau.split == "test"].reset_index(drop=True)
    acc, auc, test_loss, vrais, predits, scores = metriques_lame(lames_test, afficher_erreurs=True)
    print(f"\n===== TEST (MIL attention, niveau LAME) =====")
    print(f"Accuracy : {acc:.3f} | AUC : {auc:.3f} | loss : {test_loss:.4f}")
    matrice = confusion_matrix(vrais, predits, labels=[0, 1])
    print("Matrice de confusion (lignes=vrai, colonnes=predit) :")
    print("          predit_FH  predit_FL")
    print(f"vrai_FH      {matrice[0,0]:5d}    {matrice[0,1]:6d}")
    print(f"vrai_FL      {matrice[1,0]:5d}    {matrice[1,1]:6d}")
    print(classification_report(vrais, predits, target_names=CLASSES, digits=3))

    # --- MATRICE DE CONFUSION EN IMAGE (pour le rapport) ---
    fig2, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(matrice, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASSES); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predit"); ax.set_ylabel("Vrai")
    ax.set_title("Matrice de confusion (TEST, MIL)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, matrice[i, j], ha="center", va="center",
                    color="white" if matrice[i, j] > matrice.max()/2 else "black", fontsize=14)
    plt.tight_layout()
    chemin_matrice = DOSSIER_SORTIE / "matrice_confusion_MIL.png"
    plt.savefig(chemin_matrice, dpi=130)
    print(f"Matrice de confusion enregistree -> {chemin_matrice}")
     # --- SAUVEGARDE DU MODELE (pour les cartes d'attention) ---
    torch.save(modele.state_dict(), DOSSIER_SORTIE / "modele_MIL_Virchow2.pth")
    print(f"Modele sauvegarde -> {DOSSIER_SORTIE / 'modele_MIL_Virchow2.pth'}")


if __name__ == "__main__":
    main()