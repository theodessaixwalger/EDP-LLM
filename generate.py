import torch
from model import MonLLM
from tokeniseur import decoupage
import chemins

device = 'cuda' if torch.cuda.is_available() else 'cpu'

nb_morceaux = 200
temperature = 0.8

modele = MonLLM.charger(chemins.MODELE_BASE, device)

tok = decoupage(modele.embed_tokens.num_embeddings)

print(f"Modèle chargé sur {device}. Entrée vide pour quitter.")
while True:
    amorce = input("\nAmorce > ")
    if not amorce.strip():
        break

    amorce, inconnus = tok.nettoyer(amorce)
    if inconnus:
        print(f"Caractères inconnus du modèle, ignorés : {' '.join(inconnus)}")
    if not amorce:
        continue

    contexte = torch.tensor([tok.encoder(amorce)], dtype=torch.long, device=device)
    resultat = modele.generer(contexte, max_nouveaux_tokens=nb_morceaux, temperature=temperature)
    print()
    print(tok.decoder(resultat[0].tolist()))
