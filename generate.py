import torch
import pickle
from model import MonLLM

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Réglages de génération
nb_caracteres = 500   # longueur du texte écrit à la suite de l'amorce
temperature = 0.8     # en dessous de 1 le texte est plus sage, au dessus plus varié

# On recharge le vocabulaire figé par prepare.py : les identifiants doivent
# correspondre exactement à ceux vus pendant l'entraînement, sinon le texte
# produit serait décodé n'importe comment.
with open('meta.pkl', 'rb') as f:
    meta = pickle.load(f)
taille_vocab = meta['taille_vocab']
id_vers_car = meta['id_vers_car']
car_vers_id = meta['car_vers_id']

# Le squelette du modèle est reconstruit à vide (avec la même architecture que
# dans train.py, ici les valeurs par défaut), puis rempli avec les poids appris.
modele = MonLLM(taille_vocab)
modele.load_state_dict(torch.load('checkpoint.pt', map_location=device))
modele.to(device)
modele.eval()  # désactive le dropout : en génération on veut tout le réseau

# Mêmes conversions que dans prepare.py, mais à partir du vocabulaire rechargé
def encoder(s):
    return [car_vers_id[c] for c in s]
def decoder(l):
    return ''.join([id_vers_car[i] for i in l])

# Le chargement du modèle est la partie lente : on le fait une seule fois, puis
# la boucle permet d'essayer autant d'amorces qu'on veut sans relancer le script.
print(f"Modèle chargé sur {device}. Entrée vide pour quitter.")
while True:
    amorce = input("\nAmorce > ")
    if not amorce.strip():
        break

    # L'amorce est tapée au clavier : elle peut contenir des caractères absents
    # du corpus, que le vocabulaire ne sait donc pas encoder.
    inconnus = sorted(set(amorce) - set(car_vers_id))
    if inconnus:
        print(f"Caractères inconnus du modèle, ignorés : {' '.join(inconnus)}")
        amorce = ''.join(c for c in amorce if c in car_vers_id)
        if not amorce:
            continue

    # Le tensor a deux dimensions car le modèle attend toujours un batch,
    # ici de taille 1 (une seule séquence).
    contexte = torch.tensor([encoder(amorce)], dtype=torch.long, device=device)
    resultat = modele.generer(contexte, max_nouveaux_tokens=nb_caracteres, temperature=temperature)
    print()
    print(decoder(resultat[0].tolist()))  # l'amorce suivie du texte généré
