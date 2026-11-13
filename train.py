import math
import os
import sys
import torch
import numpy as np
import time
from model import MonLLM
from tokeniseur import Tokeniseur
import chemins

# --- Configuration ---
taille_contexte = 4096
taille_batch = 4
dim_embed = 640
nb_tetes = 10
nb_blocs = 10
dropout = 0.0
taux_apprentissage = 3e-4
max_iterations = 80000
intervalle_eval = 1000
iterations_eval = 50
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# python train.py --reprendre : repart de la dernière sauvegarde si l'entraînement a été coupé
FICHIER_REPRISE = os.path.join(chemins.MODELES, 'reprise.pt')
reprendre = '--reprendre' in sys.argv

iterations_echauffement = 1000
taux_final = taux_apprentissage / 10

# Le taux monte doucement pendant 1000 itérations puis redescend en cosinus
def taux(iteration):
    if iteration < iterations_echauffement:
        return taux_apprentissage * (iteration + 1) / iterations_echauffement
    progression = (iteration - iterations_echauffement) / (max_iterations - iterations_echauffement)
    return taux_final + 0.5 * (taux_apprentissage - taux_final) * (1 + math.cos(math.pi * progression))

print(f"Device utilisé : {device}")

# --- Chargement des données ---
# memmap : train.bin fait plus de 3 Go, on le laisse sur le disque
train_data = np.memmap(chemins.TRAIN_BIN, dtype=np.uint16, mode='r')
val_data = np.memmap(chemins.VAL_BIN, dtype=np.uint16, mode='r')
taille_vocab = Tokeniseur.charger().taille_vocab

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - taille_contexte, (taille_batch,))
    x = torch.stack([torch.from_numpy(data[i:i+taille_contexte].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+taille_contexte].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimer_perte():
    resultats = {}
    modele.eval()
    for split in ['train', 'val']:
        pertes = torch.zeros(iterations_eval)
        for k in range(iterations_eval):
            X, Y = get_batch(split)
            _, perte = modele(X, Y)
            pertes[k] = perte.item()
        resultats[split] = pertes.mean().item()
    modele.train()
    return resultats

# --- Modèle et optimiseur ---
modele = MonLLM(taille_vocab, dim_embed, nb_tetes, nb_blocs, taille_contexte, dropout)
modele = modele.to(device)
nb_params = sum(p.numel() for p in modele.parameters())
print(f"Nombre de paramètres : {nb_params/1e6:.2f}M")

optimiseur = torch.optim.AdamW(modele.parameters(), lr=taux_apprentissage)

depart = 0
if reprendre and os.path.exists(FICHIER_REPRISE):
    etat = torch.load(FICHIER_REPRISE, map_location=device, weights_only=True)
    modele.load_state_dict(etat['modele'])
    optimiseur.load_state_dict(etat['optimiseur'])
    depart = int(etat['iteration'])
    print(f"Reprise à l'itération {depart}")

# --- Boucle d'entraînement ---
debut = time.time()
for iteration in range(depart, max_iterations):
    if (iteration % intervalle_eval == 0 and iteration > depart) or iteration == max_iterations - 1 \
            or iteration == 0:
        pertes = estimer_perte()
        temps_ecoule = time.time() - debut
        print(f"step {iteration}: train loss {pertes['train']:.4f}, val loss {pertes['val']:.4f} ({temps_ecoule:.0f}s)")
        torch.save(modele.state_dict(), chemins.MODELE_BASE)
        torch.save({'modele': modele.state_dict(), 'optimiseur': optimiseur.state_dict(),
                    'iteration': iteration}, FICHIER_REPRISE)

    for groupe in optimiseur.param_groups:
        groupe['lr'] = taux(iteration)
    xb, yb = get_batch('train')
    logits, perte = modele(xb, yb)
    optimiseur.zero_grad(set_to_none=True)
    perte.backward()
    torch.nn.utils.clip_grad_norm_(modele.parameters(), 1.0)
    optimiseur.step()

if os.path.exists(FICHIER_REPRISE):
    os.remove(FICHIER_REPRISE)
print(f"Entraînement terminé. Modèle sauvegardé dans {chemins.MODELE_BASE}")
