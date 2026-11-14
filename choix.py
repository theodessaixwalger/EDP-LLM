import re
import torch
from recherche import DOCUMENT

# Questions à choix : le modèle écrit mal sa réponse (il cite un seul choix une fois sur trois) mais sait
# souvent lequel est le plus probable après avoir lu le document. Pour ces questions on ne génère pas,
# on prend le choix qu'il trouve le plus probable (57 % au QCM v2 avec documents, 43 % en écrivant)

# « Quelle est la capitale de la France ? (Lyon, Paris, Marseille ou Nice) »
LISTE = re.compile(r"^(.*?)\s*\(([^()]+)\)\s*[?.!]*\s*$", re.S)

def trouver_choix(message):
    m = LISTE.match(message)
    if not m or ' ou ' not in m.group(2):
        return None
    debut, _, dernier = m.group(2).rpartition(' ou ')
    choix = [c.strip() for c in debut.split(',')] + [dernier.strip()]
    if not m.group(1).strip() or not 2 <= len(choix) <= 6 or any(not c or len(c) > 80 for c in choix):
        return None
    return m.group(1).strip(), choix

# Log-probabilité de chaque choix comme réponse, en somme et divisée par le nombre de caractères
# (la somme avantage les choix courts, et ça dépend pas du découpage en morceaux)
@torch.no_grad()
def vraisemblance(modele, tok, contexte, choix):
    device = next(modele.parameters()).device
    fenetre = modele.taille_contexte
    debut = tok.encoder(contexte)
    suites = [tok.encoder(c + '\n') for c in choix]
    sequences = [(debut + s)[-(fenetre + 1):] for s in suites]
    longueur = max(map(len, sequences))
    entree = torch.full((len(choix), longueur - 1), tok.saut_de_ligne, dtype=torch.long, device=device)
    for i, s in enumerate(sequences):
        entree[i, :len(s) - 1] = torch.tensor(s[:-1], device=device)
    log_probas = torch.log_softmax(modele(entree)[0], dim=-1)
    sommes = []
    for i, (s, suite) in enumerate(zip(sequences, suites)):
        positions = torch.arange(len(s) - 1 - len(suite), len(s) - 1, device=device)
        cibles = torch.tensor(s[-len(suite):], device=device)
        sommes.append(log_probas[i, positions, cibles].sum().item())
    return sommes, [v / len(c + '\n') for v, c in zip(sommes, choix)]

# La phrase des documents (les lignes « Document : ... », le meilleur en dernier) qui contient le choix,
# pour montrer d'où vient la réponse. Pas les phrases dont le dump a perdu le début (« est un jeu télévisé... »)
def phrase_qui_cite(documents, choix):
    for ligne in reversed(documents.splitlines()):
        for phrase in re.split(r"(?<=[.!?])\s+", ligne.removeprefix(DOCUMENT).strip()):
            debut_propre = phrase[:1].isupper() or phrase[:1].isdigit() or phrase[:1] == '«'
            if debut_propre and choix.lower() in phrase.lower() and len(phrase) <= 300:
                return phrase.strip()
    return None

def repondre(modele, tok, contexte, choix, documents):
    _, normalisees = vraisemblance(modele, tok, contexte, choix)
    meilleur = choix[max(range(len(choix)), key=normalisees.__getitem__)]
    reponse = meilleur[:1].upper() + meilleur[1:].rstrip('.') + '.'
    phrase = phrase_qui_cite(documents, meilleur)
    return reponse + ' ' + phrase if phrase else reponse
