import json
import os
import torch
import torch.nn.functional as F
from model import MonLLM
from tokeniseur import decoupage
import chemins

FICHIER_PREFERENCES = chemins.PREFERENCES
UTILISATEUR = "Utilisateur : "
ASSISTANT = "Assistant : "
longueur_max_reponse = 400

TEMPERATURES = (0.6, 1.0)

SUGGESTIONS = [
    "salut", "comment ça va ?", "ça va aussi", "je vais bien", "et toi ?",
    "raconte moi une blague", "c'est quoi la photosynthèse ?", "tu peux m'aider en maths ?",
    "ok", "mdr", "je m'appelle Théo", "comment je m'appelle", "quel âge j'ai",
]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

fichier_modele = chemins.modele_le_plus_recent()
modele = MonLLM.charger(fichier_modele, device)
taille_contexte = modele.taille_contexte
tok = decoupage(modele.embed_tokens.num_embeddings)

@torch.no_grad()
def repondre(contexte, temperature):
    idx = torch.tensor([tok.encoder(contexte)], dtype=torch.long, device=device)
    reponse = ''
    for _ in range(longueur_max_reponse):
        logits, _ = modele(idx[:, -taille_contexte:])
        probabilites = F.softmax(logits[:, -1, :] / temperature, dim=-1)
        suivant = torch.multinomial(probabilites, num_samples=1)
        if suivant.item() == tok.saut_de_ligne:
            break
        reponse += tok.morceaux[suivant.item()]
        idx = torch.cat((idx, suivant), dim=1)
    return reponse

def enregistrer(contexte, choisie, rejetee):
    if choisie.strip() == rejetee.strip():
        return False
    with open(FICHIER_PREFERENCES, 'a', encoding='utf-8') as f:
        f.write(json.dumps({'contexte': contexte, 'choisie': choisie,
                            'rejetee': rejetee, 'origine': 'manuel'}, ensure_ascii=False) + '\n')
    return True

def nettoyer(message):
    message, inconnus = tok.nettoyer(message)
    if inconnus:
        print(f"Caractères inconnus du modèle, ignorés : {' '.join(inconnus)}")
    return message

deja = sum(1 for _ in open(FICHIER_PREFERENCES, encoding='utf-8')) if os.path.exists(FICHIER_PREFERENCES) else 0
print(f"Modèle : {fichier_modele} — {deja} préférences déjà enregistrées")
print("Pour chaque message, deux réponses sont proposées. Indiquez la meilleure.")
print("  a / b : la bonne réponse       c : aucune des deux, je l'écris moi-même")
print("  p : passer   n : nouvelle conversation   entrée vide : quitter")
print(f"\nSuggestions : {', '.join(SUGGESTIONS[:6])}...")

historique = ''
enregistrees = 0

while True:
    message = input(f"\n{'(nouvelle conversation) ' if not historique else ''}Vous > ").strip()
    if not message:
        break
    if message == 'n':
        historique = ''
        print("Conversation remise à zéro.")
        continue

    message = nettoyer(message)
    if not message:
        continue

    contexte = historique + UTILISATEUR + message + '\n' + ASSISTANT
    propositions = [repondre(contexte, t) for t in TEMPERATURES]
    print(f"  a) {propositions[0]}")
    print(f"  b) {propositions[1]}")

    choix = input("  meilleure ? [a/b/c/p] > ").strip().lower()
    if choix == 'p':
        continue
    if choix == 'c':
        correction = nettoyer(input("  réponse attendue > ").strip())
        if not correction:
            continue
        for proposition in propositions:
            enregistrees += enregistrer(contexte, correction, proposition)
        retenue = correction
    elif choix in ('a', 'b'):
        retenue = propositions[0] if choix == 'a' else propositions[1]
        autre = propositions[1] if choix == 'a' else propositions[0]
        enregistrees += enregistrer(contexte, retenue, autre)
    else:
        continue

    historique = contexte + retenue + '\n'
    if len(tok.encoder(historique)) > taille_contexte:
        historique = ''
        print("  (conversation trop longue, on repart de zéro)")

print(f"\n{enregistrees} préférences ajoutées à {FICHIER_PREFERENCES}")
