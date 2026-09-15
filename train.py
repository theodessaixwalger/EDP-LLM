import torch
import numpy as np
import pickle
import time
from model import MonLLM

# --- Configuration ---
# Ces valeurs définissent la taille du réseau : elles doivent rester identiques
# pour recharger le checkpoint dans generate.py.
taille_contexte = 256   # nombre de caractères que le modèle voit d'un coup
taille_batch = 64       # séquences traitées en parallèle à chaque itération
dim_embed = 384
nb_tetes = 6
nb_blocs = 6
dropout = 0.2           # neurones désactivés au hasard, contre le surapprentissage
taux_apprentissage = 3e-4
max_iterations = 5000
intervalle_eval = 250   # fréquence des mesures de perte et des sauvegardes
iterations_eval = 200   # batchs moyennés à chaque mesure
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device utilisé : {device}")

# --- Chargement des données ---
# memmap : les fichiers restent sur le disque, seuls les morceaux tirés sont
# chargés en mémoire. Indispensable ici, train.bin fait ~30 Mo.
train_data = np.memmap('train.bin', dtype=np.uint16, mode='r')
val_data = np.memmap('val.bin', dtype=np.uint16, mode='r')
# Le vocabulaire vient de prepare.py : la taille de la couche de sortie en dépend.
with open('meta.pkl', 'rb') as f:
    meta = pickle.load(f)
taille_vocab = meta['taille_vocab']

def get_batch(split):
    """Tire au hasard taille_batch extraits du corpus."""
    data = train_data if split == 'train' else val_data
    # Positions de départ aléatoires, choisies assez tôt dans le fichier pour
    # qu'il reste toujours une séquence complète à lire derrière.
    ix = torch.randint(len(data) - taille_contexte, (taille_batch,))
    x = torch.stack([torch.from_numpy(data[i:i+taille_contexte].astype(np.int64)) for i in ix])
    # y est x décalé d'un cran : la cible d'une position est le caractère suivant
    y = torch.stack([torch.from_numpy(data[i+1:i+1+taille_contexte].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimer_perte():
    """Moyenne la perte sur plusieurs batchs pour lisser le bruit d'un tirage
    isolé. Un écart qui se creuse entre val et train signale du surapprentissage."""
    resultats = {}
    modele.eval()  # désactive le dropout : on mesure, on n'entraîne pas
    for split in ['train', 'val']:
        pertes = torch.zeros(iterations_eval)
        for k in range(iterations_eval):
            X, Y = get_batch(split)
            _, perte = modele(X, Y)
            pertes[k] = perte.item()
        resultats[split] = pertes.mean().item()
    modele.train()  # retour en mode entraînement
    return resultats

# --- Modèle et optimiseur ---
modele = MonLLM(taille_vocab, dim_embed, nb_tetes, nb_blocs, taille_contexte, dropout)
modele = modele.to(device)
nb_params = sum(p.numel() for p in modele.parameters())
print(f"Nombre de paramètres : {nb_params/1e6:.2f}M")

optimiseur = torch.optim.AdamW(modele.parameters(), lr=taux_apprentissage)

# --- Boucle d'entraînement ---
debut = time.time()
for iteration in range(max_iterations):

    # Point de contrôle régulier : on affiche l'avancement et on sauvegarde, ce
    # qui permet de reprendre le modèle même si l'entraînement est interrompu.
    if iteration % intervalle_eval == 0:
        pertes = estimer_perte()
        temps_ecoule = time.time() - debut
        print(f"step {iteration}: train loss {pertes['train']:.4f}, val loss {pertes['val']:.4f} ({temps_ecoule:.0f}s)")
        torch.save(modele.state_dict(), 'checkpoint.pt')

    # Une itération = un batch : prédiction, mesure de l'erreur, correction des poids
    xb, yb = get_batch('train')
    logits, perte = modele(xb, yb)
    optimiseur.zero_grad(set_to_none=True)  # sans remise à zéro les gradients s'additionnent
    perte.backward()                        # calcule le gradient de l'erreur
    optimiseur.step()                       # ajuste les poids dans le sens opposé

print("Entraînement terminé. Modèle sauvegardé dans checkpoint.pt")
