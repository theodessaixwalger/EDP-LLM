import torch
import numpy as np
import time
from model import MonLLM
import chemins

# --- Configuration ---
taille_contexte = 4096
taille_batch = 4
dropout = 0.1
taux_apprentissage = 1e-4
max_iterations = 3000
intervalle_eval = 250
iterations_eval = 50
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device utilisé : {device}")

# --- Chargement des données ---
donnees = {}
for split in ['train', 'val']:
    fichier = np.load(chemins.CHAT_TRAIN if split == 'train' else chemins.CHAT_VAL)
    masques = {'masque': torch.from_numpy(fichier['masque'])}
    if split == 'val':
        masques['memoire'] = torch.from_numpy(fichier['memoire'])
        masques['document'] = torch.from_numpy(fichier['document'])
    donnees[split] = (torch.from_numpy(fichier['ids'].astype(np.int16)), masques)

def get_batch(split, cible='masque'):
    ids, masques = donnees[split]
    masque = masques[cible]
    ix = torch.randint(len(ids), (taille_batch,))
    lignes = ids[ix].long()
    x = lignes[:, :-1]
    y = lignes[:, 1:].clone()
    # -100 = ignoré par la perte : on apprend que les réponses de l'assistant, pas ce que tape l'utilisateur
    y[~masque[ix, 1:]] = -100
    return x.to(device), y.to(device)

@torch.no_grad()
def estimer_perte():
    resultats = {}
    modele.eval()
    for nom, split, cible in (('train', 'train', 'masque'), ('val', 'val', 'masque'),
                              ('memoire', 'val', 'memoire'), ('document', 'val', 'document')):
        pertes = torch.zeros(iterations_eval)
        for k in range(iterations_eval):
            X, Y = get_batch(split, cible)
            _, perte = modele(X, Y)
            pertes[k] = perte.item()
        resultats[nom] = pertes.nanmean().item()
    modele.train()
    return resultats

# --- Reprise du modèle de l'étape 1 ---
etat = torch.load(chemins.MODELE_BASE, map_location='cpu', weights_only=True)
architecture = MonLLM.architecture(etat)
architecture['taille_contexte'] = taille_contexte
modele = MonLLM(**architecture, dropout=dropout)
etat['fenetre'] = modele.fenetre
etat.setdefault('nb_tetes', modele.nb_tetes)
modele.load_state_dict(etat)
modele = modele.to(device)
nb_params = sum(p.numel() for p in modele.parameters())
print(f"Nombre de paramètres : {nb_params/1e6:.2f}M")

optimiseur = torch.optim.AdamW(modele.parameters(), lr=taux_apprentissage)

# --- Boucle de fine-tuning ---
meilleur_score = float('inf')
debut = time.time()
for iteration in range(max_iterations):
    if iteration % intervalle_eval == 0 or iteration == max_iterations - 1:
        pertes = estimer_perte()
        temps_ecoule = time.time() - debut
        print(f"step {iteration}: train loss {pertes['train']:.4f}, val loss {pertes['val']:.4f}, "
              f"mémoire {pertes['memoire']:.4f}, documents {pertes['document']:.4f} ({temps_ecoule:.0f}s)")
        # On garde le modèle qui retrouve le mieux les faits et les documents, pas celui qui a la
        # meilleure perte globale (elle remonte très vite à cause des répliques de romans)
        if pertes['memoire'] + pertes['document'] < meilleur_score:
            meilleur_score = pertes['memoire'] + pertes['document']
            torch.save(modele.state_dict(), chemins.MODELE_CHAT)

    xb, yb = get_batch('train')
    logits, perte = modele(xb, yb)
    optimiseur.zero_grad(set_to_none=True)
    perte.backward()
    torch.nn.utils.clip_grad_norm_(modele.parameters(), 1.0)
    optimiseur.step()

print(f"Fine-tuning terminé. Meilleure perte mémoire + documents : {meilleur_score:.4f}, modèle sauvegardé dans {chemins.MODELE_CHAT}")
