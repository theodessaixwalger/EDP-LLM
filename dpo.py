import copy
import json
import random
import time
import torch
import torch.nn.functional as F
from model import MonLLM
from tokeniseur import decoupage
import chemins

random.seed(1337)
torch.manual_seed(1337)

# --- Configuration ---
FICHIER_PREFERENCES = chemins.PREFERENCES
MODELE_DEPART = chemins.MODELE_CHAT
MODELE_SORTIE = chemins.MODELE_DPO

taille_batch = 8
taux_apprentissage = 1e-5
epoques = 4

# beta : à quel point le modèle a le droit de s'éloigner de la référence
beta = 0.1

# On garde un peu d'imitation de la bonne réponse, sinon il finit par tout refuser
poids_sft = 0.3

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device utilisé : {device}")

# --- Le modèle entraîné, et sa copie figée ---
modele = MonLLM.charger(MODELE_DEPART, device).train()
taille_contexte = modele.taille_contexte
tok = decoupage(modele.embed_tokens.num_embeddings)

# Copie figée du modèle de départ, sert de point de comparaison
reference = copy.deepcopy(modele)
reference.eval()
for parametre in reference.parameters():
    parametre.requires_grad = False

# --- Données de préférence ---
def encoder(contexte, reponse):
    ids_reponse = tok.encoder(reponse + '\n')
    ids_contexte = tok.encoder(contexte)
    place = taille_contexte - len(ids_reponse)
    ids_contexte = ids_contexte[-place:] if place > 0 else []
    return ids_contexte + ids_reponse, [0] * len(ids_contexte) + [1] * len(ids_reponse)

paires = []
with open(FICHIER_PREFERENCES, 'r', encoding='utf-8') as f:
    for ligne in f:
        exemple = json.loads(ligne)
        if exemple['choisie'].strip() == exemple['rejetee'].strip():
            continue
        paires.append((encoder(exemple['contexte'], exemple['choisie']),
                       encoder(exemple['contexte'], exemple['rejetee'])))

random.shuffle(paires)
coupure = max(1, int(len(paires) * 0.1))
paires_val, paires_train = paires[:coupure], paires[coupure:]
print(f"{len(paires):,} paires de préférence ({len(paires_train)} entraînement, {len(paires_val)} validation)")

def en_tenseurs(lot):
    longueur = max(len(ids) for paire in lot for ids, _ in paire)
    remplissage = tok.saut_de_ligne

    def tenseurs(cote):
        ids = [p[cote][0] + [remplissage] * (longueur - len(p[cote][0])) for p in lot]
        masques = [p[cote][1] + [0] * (longueur - len(p[cote][1])) for p in lot]
        return (torch.tensor(ids, device=device),
                torch.tensor(masques, dtype=torch.float, device=device))

    choisies, masques_c = tenseurs(0)
    rejetees, masques_r = tenseurs(1)
    return choisies, masques_c, rejetees, masques_r

def log_probas(reseau, sequences, masques):
    logits, _ = reseau(sequences[:, :-1])
    cibles = sequences[:, 1:]
    detail = torch.log_softmax(logits, dim=-1).gather(-1, cibles.unsqueeze(-1)).squeeze(-1)
    return (detail * masques[:, 1:]).sum(dim=1)

def pertes_du_lot(lot):
    choisies, masques_c, rejetees, masques_r = en_tenseurs(lot)

    lp_choisie = log_probas(modele, choisies, masques_c)
    lp_rejetee = log_probas(modele, rejetees, masques_r)
    with torch.no_grad():
        ref_choisie = log_probas(reference, choisies, masques_c)
        ref_rejetee = log_probas(reference, rejetees, masques_r)

    # DPO : la bonne réponse doit gagner en probabilité plus que la mauvaise, par rapport à la référence
    marge = (lp_choisie - ref_choisie) - (lp_rejetee - ref_rejetee)
    perte_dpo = -F.logsigmoid(beta * marge).mean()
    perte_sft = -(lp_choisie / masques_c.sum(dim=1)).mean()
    return perte_dpo + poids_sft * perte_sft, (marge > 0).float().mean().item()

@torch.no_grad()
def evaluer():
    modele.eval()
    justes = 0
    for depart in range(0, len(paires_val), taille_batch):
        _, part = pertes_du_lot(paires_val[depart:depart + taille_batch])
        justes += part * len(paires_val[depart:depart + taille_batch])
    modele.train()
    return justes / len(paires_val)

# --- Entraînement ---
optimiseur = torch.optim.AdamW(modele.parameters(), lr=taux_apprentissage)
print(f"Accord avec les préférences avant entraînement : {evaluer():.1%}")

debut = time.time()
for epoque in range(epoques):
    random.shuffle(paires_train)
    total, lots = 0, 0
    for depart in range(0, len(paires_train), taille_batch):
        lot = paires_train[depart:depart + taille_batch]
        perte, _ = pertes_du_lot(lot)
        optimiseur.zero_grad(set_to_none=True)
        perte.backward()
        optimiseur.step()
        total += perte.item()
        lots += 1
    print(f"époque {epoque + 1}/{epoques} : perte {total/lots:.4f}, "
          f"accord validation {evaluer():.1%} ({time.time()-debut:.0f}s)")

torch.save(modele.state_dict(), MODELE_SORTIE)
print(f"Terminé : modèle sauvegardé dans {MODELE_SORTIE}")
