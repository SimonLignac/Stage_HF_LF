

from pathlib import Path      
import h5py                     
import numpy as np              
import pandas as pd             
import torch                   
import torch.nn as nn        
from torch.utils.data import Dataset, DataLoader   # pour livrer les donnees au modele
from sklearn.metrics import (confusion_matrix, accuracy_score,  
                             classification_report, roc_auc_score)
import matplotlib               
matplotlib.use("Agg")            # mode "sans fenetre" : on sauvegarde en image au lieu d'afficher
import matplotlib.pyplot as plt  


# config

DOSSIER_SORTIE = Path("/mnt/d/internship/output")  
MANIFEST = DOSSIER_SORTIE / "manifest_slides.csv"    
ENCODEUR = "resnet50"     
DIM_FEATURES = 2048         
CLASSES = ["FH", "FL"]   
MAX_PATCHS_PAR_LAME = 500    
NB_EPOCHS = 8                 
BATCH_SIZE = 8192         
TAUX_APPRENTISSAGE = 1e-4     #  vitesse d'apprentissage
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"   # GPU si dispo, sinon CPU
SEED = 32                  # seed du hasard (pour la reproductibilite)

torch.manual_seed(SEED); np.random.seed(SEED)      # applique la seed
label_vers_indice = {"FH": 0, "FL": 1} 
generateur = np.random.default_rng(SEED)             # generateur de hasard pour l'echantillonnage


# FONCTION : charger un echantillon de features (train ou val) en gros crée un fichier f (commet tous les patchs sont reliés dans un seul fichier par lame)

def charger_echantillon(sous_tableau):
    X, y = [], []                                    # X = features, y = labels
    for _, r in sous_tableau.iterrows():             # pour chaque lame (r) du groupe
        with h5py.File(r["h5_path"], "r") as f:      # ouvre son fichier H5 le "r" veut dire read 
            feats = f[f"features/{ENCODEUR}"][...]   # lit tous ses vecteurs de features et f permet d'insérer la valuer d'une variable dans le texte ici encode c'est H-optimus. le [...]veut dire lit toutes les valeurs et mettre dans un tableau numpy
        if len(feats) > MAX_PATCHS_PAR_LAME:         # si trop de patches...
            idx = generateur.choice(len(feats), MAX_PATCHS_PAR_LAME, replace=False)  # tire 500 au hasard
            feats = feats[idx]                        # garde ceux-la
        X.append(feats)                               # ajoute les features
        y.append(np.full(len(feats), label_vers_indice[r["label"]]))  # 1 label par patch (celui de la lame)
    return np.concatenate(X), np.concatenate(y)       # colle tout en 2 grands tableaux



# CLASSE : en gros on passe les données en dataset pour qu'elles puissent être traitées par Pytorch

class JeuDePatchs(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()          # features en tenseur decimal
        self.y = torch.from_numpy(y).long()           # labels en tenseur entier
    def __len__(self):                                # nombre d'exemples
        return len(self.y)
    def __getitem__(self, i):                         # renvoie l'exemple i (patch + label)
        return self.X[i], self.y[i]



# CLASSE : le modele 

class ReseauPatch(nn.Module):
    def __init__(self, dim_entree, nb_classes):
        super().__init__()                            # initialisation PyTorch obligatoire
        self.reseau = nn.Sequential(                  # empile les couches dans l'ordre
            nn.Linear(dim_entree, 512),   # transforme 1536 -> 512
            nn.ReLU(),                # non-linearite (garde le positif, annule le negatif)
            nn.Dropout(0.3),          # eteint 30% des neurones (anti-surapprentissage)
            nn.Linear(512, 128),      # transforme 512 -> 128
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, nb_classes),  # transforme 128 -> 2 scores (FH, FL)
        )
    def forward(self, x):                             # calcul quand on passe des donnees
        return self.reseau(x)



# FONCTION PRINCIPALE

def main():
    tableau = pd.read_csv(MANIFEST)                   # lit le manifest et après on part de ça pour avoir sous_tableau avec panda : tableau [tableau]
    print(f"{len(tableau)} lames | splits : {tableau['split'].value_counts().to_dict()}")

    # On charge les donnees train et validation 
    print("Chargement train...")
    X_train, y_train = charger_echantillon(tableau[tableau.split == "train"])  # features des lames "train"
    print("Chargement val...")
    X_val, y_val = charger_echantillon(tableau[tableau.split == "val"])        # features des lames "val"
    print(f"Patches : train {len(y_train)}, val {len(y_val)}")

    # DataLoader = on prépare les données en petit paquet au lieu de tout d'un coup en gros 
    chargeur_train = DataLoader(JeuDePatchs(X_train, y_train), batch_size=BATCH_SIZE,
                                shuffle=True, num_workers=12)   # train : melange
    chargeur_val = DataLoader(JeuDePatchs(X_val, y_val), batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=12)    # val : pas besoin de melanger

    modele = ReseauPatch(DIM_FEATURES, len(CLASSES)).to(DEVICE)  # cree le modele, l'envoie sur GPU
    optimiseur = torch.optim.AdamW(modele.parameters(), lr=TAUX_APPRENTISSAGE, weight_decay=1e-2)  # l'optimiseur (ajuste les poids)
    critere = nn.CrossEntropyLoss()                          # la fonction de perte (mesure l'erreur)

    # Dictionnaire pour garder l'historique (1 valeur par epoch)
    historique = {"train_loss": [], "val_loss": [], "val_acc": [], "val_auc": []}

    # Fonction interne : mesure acc et AUC au niveau LAME (vote majoritaire) 
    @torch.no_grad()                                  # pas de gradients : on ne fait que mesurer
    def metriques_lame(sous_tableau):
        modele.eval()                                 # mode evaluation (Dropout desactive)
        vrais, predits, scores = [], [], []
        for _, r in sous_tableau.iterrows():          # pour chaque lame
            with h5py.File(r["h5_path"], "r") as f:
                feats = f[f"features/{ENCODEUR}"][...]
            probas = []
            for i in range(0, len(feats), BATCH_SIZE):  # par paquets
                xb = torch.from_numpy(feats[i:i+BATCH_SIZE]).float().to(DEVICE)
                probas.append(torch.softmax(modele(xb), 1).cpu().numpy())  # scores -> probabilites
            probas = np.concatenate(probas)
            vote = np.bincount(probas.argmax(1), minlength=2).argmax()  # vote majoritaire des patches avec bicount 
            vrais.append(label_vers_indice[r["label"]])   # vrai label
            predits.append(vote)                       # label predit
            scores.append(probas[:, 1].mean())         # proba moyenne FL (pour l'AUC)
        vrais = np.array(vrais); predits = np.array(predits)
        acc = accuracy_score(vrais, predits)           # calcule l'accuracy
        try:    auc = roc_auc_score(vrais, scores)      # calcule l'AUC
        except ValueError: auc = float("nan")           # si une seule classe, AUC impossible
        return acc, auc, vrais, predits, scores

    #  Fonction interne : loss moyenne sur un chargeur (sans apprendre) 
    @torch.no_grad()
    def calculer_loss(chargeur):
        modele.eval(); total = 0.0; n = 0 #model eval desactive le dropout , n c'est le nb de patchs traités et total accumule l'erreur
        for xb, yb in chargeur:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            total += critere(modele(xb), yb).item() * len(yb); n += len(yb)  # accumule l'erreur .item extrait la valeur numerique de la loss, fois len on multiplie par le nombre de patchs dans le paquet
        return total / n                               # moyenne

    #  LA BOUCLE D'ENTRAINEMENT 
    print("\nEntrainement :")
    for epoch in range(1, NB_EPOCHS + 1):             # repete NB_EPOCHS fois
        modele.train(); total = 0.0                    # mode entrainement (Dropout actif)
        for xb, yb in chargeur_train:                  # parcourt les paquets
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)      # envoie sur GPU
            optimiseur.zero_grad()                     # nettoie les gradients
            loss = critere(modele(xb), yb)             # calcule l'erreur du paquet
            loss.backward()                            # calcule comment corriger les poids
            optimiseur.step()                          # applique la correction (apprentissage)
            total += loss.item() * len(yb)             # accumule la perte
        train_loss = total / len(y_train)              # perte moyenne d'entrainement

        # mesures sur la validation a la fin de l'epoch
        val_loss = calculer_loss(chargeur_val)         # perte de validation
        val_acc, val_auc, *_ = metriques_lame(tableau[tableau.split == "val"])  # acc et AUC de validation

        # on stocke tout dans l'historique
        historique["train_loss"].append(train_loss)
        historique["val_loss"].append(val_loss)
        historique["val_acc"].append(val_acc)
        historique["val_auc"].append(val_auc)
        print(f"  epoch {epoch:2d}/{NB_EPOCHS}  train_loss {train_loss:.4f} | "
              f"val_loss {val_loss:.4f} | val_acc {val_acc:.3f} | val_auc {val_auc:.3f}")

    # --- TRACE DES 3 COURBES ---
    liste_epochs = range(1, NB_EPOCHS + 1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))   # 3 graphiques cote a cote

    # Graphe 1 : loss train vs val (le graphe cle pour le surapprentissage)
    axes[0].plot(liste_epochs, historique["train_loss"], label="train", marker="o", ms=3)
    axes[0].plot(liste_epochs, historique["val_loss"],   label="validation", marker="o", ms=3)
    axes[0].set_title("Loss (entrainement vs validation)")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss"); axes[0].legend(); axes[0].grid(alpha=.3)

    # Graphe 2 : accuracy de validation
    axes[1].plot(liste_epochs, historique["val_acc"], color="green", marker="o", ms=3)
    axes[1].set_title("Accuracy validation (niveau lame)")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("accuracy"); axes[1].grid(alpha=.3)

    # Graphe 3 : AUC de validation
    axes[2].plot(liste_epochs, historique["val_auc"], color="purple", marker="o", ms=3)
    axes[2].set_title("AUC validation (niveau lame)")
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("AUC"); axes[2].grid(alpha=.3)

    plt.tight_layout()                                 # ajuste les espacements
    chemin_courbes = DOSSIER_SORTIE / "courbes_apprentissage.png"
    plt.savefig(chemin_courbes, dpi=130)               # sauvegarde l'image
    print(f"\nCourbes enregistrees -> {chemin_courbes}")

    # --- EVALUATION FINALE SUR LE TEST ---
    acc, auc, vrais, predits, scores = metriques_lame(tableau[tableau.split == "test"])
    print(f"\n===== TEST (niveau LAME, vote majoritaire) =====")
    print(f"Accuracy : {acc:.3f} | AUC : {auc:.3f}")
    matrice = confusion_matrix(vrais, predits, labels=[0, 1])   # matrice de confusion
    print("Matrice de confusion (lignes=vrai, colonnes=predit) :")
    print("          predit_FH  predit_FL")
    print(f"vrai_FH      {matrice[0,0]:5d}    {matrice[0,1]:6d}")
    print(f"vrai_FL      {matrice[1,0]:5d}    {matrice[1,1]:6d}")
    print(classification_report(vrais, predits, target_names=CLASSES, digits=3))

    # --- MATRICE DE CONFUSION EN IMAGE (pour le rapport) ---
    fig2, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(matrice, cmap="Blues")              # affiche la matrice en couleurs
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASSES); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predit"); ax.set_ylabel("Vrai")
    ax.set_title("Matrice de confusion (TEST)")
    for i in range(2):                                 # ecrit les chiffres dans les cases
        for j in range(2):
            ax.text(j, i, matrice[i, j], ha="center", va="center",
                    color="white" if matrice[i, j] > matrice.max()/2 else "black", fontsize=14)
    plt.tight_layout()
    chemin_matrice = DOSSIER_SORTIE / "matrice_confusion_test.png"
    plt.savefig(chemin_matrice, dpi=130)
    print(f"Matrice de confusion enregistree -> {chemin_matrice}")


# Si on lance ce fichier directement, on execute main()
if __name__ == "__main__":
    main()
