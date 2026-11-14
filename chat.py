import json
import os
import torch
import torch.nn.functional as F
from choix import repondre as repondre_par_choix, trouver_choix
from model import MonLLM, bloquer_repetitions
from outils import utiliser as utiliser_outil
from recherche import Recherche, ligne_document
from tokeniseur import decoupage
import chemins

device = 'cuda' if torch.cuda.is_available() else 'cpu'

fichier_memoire = chemins.MEMOIRE
fichier_preferences = chemins.PREFERENCES
longueur_max_reponse = 400
temperature = 0.5

temperature_seconde_chance = 0.9

UTILISATEUR = "Utilisateur : "
ASSISTANT = "Assistant : "

fichier_modele = chemins.modele_le_plus_recent()

modele = MonLLM.charger(fichier_modele, device)
taille_contexte = modele.taille_contexte
tok = decoupage(modele.embed_tokens.num_embeddings)

def longueur(texte):
    return len(tok.encoder(texte))

recherche = Recherche() if os.path.exists(os.path.join(chemins.RECHERCHE, 'positions.npy')) else None

def chercher_documents(message):
    outil = utiliser_outil(message)
    if outil:
        nom, resultat = outil
        return ligne_document(tok.nettoyer(resultat)[0]), [nom]
    if recherche is None:
        return '', []
    trouves = recherche.documents_pour(message)
    texte = ''.join(ligne_document(tok.nettoyer(passage)[0]) for _, passage in trouves)
    return texte, [titre for titre, _ in reversed(trouves)]

# --- Mémoire ---
def charger_memoire():
    if os.path.exists(fichier_memoire):
        with open(fichier_memoire, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def sauvegarder_memoire(memoire):
    with open(fichier_memoire, 'w', encoding='utf-8') as f:
        json.dump(memoire, f, ensure_ascii=False, indent=2)

def historique_visible(memoire, place):
    texte, nb, occupe = '', 0, 0
    for echange in reversed(memoire):
        bloc = UTILISATEUR + echange['utilisateur'] + '\n' + ASSISTANT + echange['assistant'] + '\n'
        taille = longueur(bloc)
        if occupe + taille > place:
            break
        texte, nb, occupe = bloc + texte, nb + 1, occupe + taille
    return texte, nb

@torch.no_grad()
def repondre(contexte, chaleur=temperature):
    idx = torch.tensor([tok.encoder(contexte)], dtype=torch.long, device=device)
    reponse, ids_reponse = '', []
    for _ in range(longueur_max_reponse):
        logits, _ = modele(idx[:, -taille_contexte:])
        logits = bloquer_repetitions(logits[0, -1], ids_reponse, tok.anti_boucle)
        suivant = torch.multinomial(F.softmax(logits / chaleur, dim=-1), num_samples=1)
        if suivant.item() == tok.saut_de_ligne:
            break
        morceau = tok.morceaux[suivant.item()]
        print(morceau, end='', flush=True)
        reponse += morceau
        ids_reponse.append(suivant.item())
        idx = torch.cat((idx, suivant.view(1, 1)), dim=1)
    print()
    return reponse

# --- Signaler une mauvaise réponse ---
def enregistrer_preference(contexte, choisie, rejetee):
    if choisie.strip() == rejetee.strip():
        return False
    with open(fichier_preferences, 'a', encoding='utf-8') as f:
        f.write(json.dumps({'contexte': contexte, 'choisie': choisie,
                            'rejetee': rejetee, 'origine': 'chat'}, ensure_ascii=False) + '\n')
    return True

def nettoyer(message):
    message, inconnus = tok.nettoyer(message)
    if inconnus:
        print(f"Caractères inconnus du modèle, ignorés : {' '.join(inconnus)}")
    return message

# --- Boucle de discussion ---
memoire = charger_memoire()
print(f"Modèle chargé sur {device} ({fichier_modele}), {len(memoire)} échanges en mémoire.")
print("Commandes : /memoire pour voir ce que le modèle voit, /mauvais pour corriger sa dernière")
print("réponse, /oublier pour tout effacer, entrée vide pour quitter.")

dernier_contexte = dernière_reponse = None

while True:
    message = input("\nVous > ").strip()
    if not message:
        break

    if message == '/mauvais':
        if dernier_contexte is None:
            print("Rien à corriger pour l'instant.")
            continue
        print("IA  > ", end='', flush=True)
        autre = repondre(dernier_contexte, temperature_seconde_chance)
        choix = input("  mieux ? [o = oui / c = j'écris la bonne réponse / n = non] > ").strip().lower()
        if choix == 'c':
            autre = nettoyer(input("  réponse attendue > ").strip())
        elif choix != 'o':
            print("  Rien enregistré.")
            continue
        if autre and enregistrer_preference(dernier_contexte, autre, dernière_reponse):
            memoire[-1]['assistant'] = autre
            sauvegarder_memoire(memoire)
            dernière_reponse = autre
            print(f"  Préférence enregistrée dans {fichier_preferences}.")
        continue

    if message == '/oublier':
        memoire.clear()
        sauvegarder_memoire(memoire)
        print("Mémoire effacée.")
        continue

    if message == '/memoire':
        texte, nb = historique_visible(memoire, taille_contexte - longueur_max_reponse)
        print(f"{len(memoire)} échanges enregistrés, dont {nb} dans la fenêtre du modèle :\n")
        print(texte, end='')
        continue

    message = nettoyer(message)
    if not message:
        continue

    question = UTILISATEUR + message + '\n' + ASSISTANT
    # Question à choix : on cherche avec la question seule et on prend le choix le plus probable
    qcm = trouver_choix(message)
    documents, sources = chercher_documents(qcm[0] if qcm else message)
    historique, _ = historique_visible(memoire, taille_contexte - longueur_max_reponse
                                       - longueur(documents + question))
    print("IA  > ", end='', flush=True)
    dernier_contexte = historique + documents + question
    if qcm:
        sans_la_liste = historique + documents + UTILISATEUR + qcm[0] + '\n' + ASSISTANT
        reponse = dernière_reponse = repondre_par_choix(modele, tok, sans_la_liste, qcm[1], documents)
        print(reponse)
    else:
        reponse = dernière_reponse = repondre(dernier_contexte)
    if sources:
        print(f"      (sources : {', '.join(sources)})")

    memoire.append({'utilisateur': message, 'assistant': reponse})
    sauvegarder_memoire(memoire)
