import json
import random
import re
import torch
import torch.nn.functional as F
from model import MonLLM
from tokeniseur import decoupage
import chemins

random.seed(1337)

FICHIER_PREFERENCES = chemins.PREFERENCES
UTILISATEUR = "Utilisateur : "
ASSISTANT = "Assistant : "
NB_SCENARIOS = 600
longueur_max_reponse = 400
temperature = 0.7

PRENOMS = ["Théo", "Julie", "Lucas", "Emma", "Hugo", "Léa", "Nathan", "Chloé", "Louis", "Manon"]
VILLES = ["Brest", "Lyon", "Nantes", "Bordeaux", "Lille", "Rennes", "Dijon", "Angers"]
METIERS = ["boulanger", "infirmier", "développeur", "professeur", "plombier", "cuisinier"]
COULEURS = ["rouge", "bleu", "vert", "jaune", "violet", "noir", "rose", "gris"]

FAITS = {
    'prénom':  {'valeur': lambda: random.choice(PRENOMS),
                'dire': "je m'appelle {v}", 'accuse': "Enchanté, {v} !",
                'question': "comment je m'appelle", 'reponse': "Votre prénom est {v}."},
    'ville':   {'valeur': lambda: random.choice(VILLES),
                'dire': "j'habite à {v}", 'accuse': "C'est noté, vous habitez à {v}.",
                'question': "où j'habite", 'reponse': "Vous habitez à {v}."},
    'âge':     {'valeur': lambda: str(random.randint(18, 80)),
                'dire': "j'ai {v} ans", 'accuse': "C'est noté, {v} ans.",
                'question': "quel âge j'ai", 'reponse': "Vous avez {v} ans."},
    'métier':  {'valeur': lambda: random.choice(METIERS),
                'dire': "je travaille comme {v}", 'accuse': "Le métier de {v}, c'est passionnant.",
                'question': "quel est mon métier", 'reponse': "Vous travaillez comme {v}."},
    'animal':  {'valeur': lambda: random.choice(PRENOMS),
                'dire': "j'ai un chat qui s'appelle {v}", 'accuse': "C'est un joli nom pour un chat, {v}.",
                'question': "comment s'appelle mon chat", 'reponse': "Votre chat s'appelle {v}."},
    'couleur': {'valeur': lambda: random.choice(COULEURS),
                'dire': "ma couleur préférée est le {v}", 'accuse': "Le {v}, bon choix !",
                'question': "quelle est ma couleur préférée", 'reponse': "Votre couleur préférée est le {v}."},
}

REFUS = ["Vous ne me l'avez pas encore dit.", "Je ne sais pas, vous ne me l'avez pas dit."]
AVEUX = ["Je ne saurais pas vous répondre.", "Je ne sais pas expliquer ça.",
         "Désolé, c'est au delà de ce que je sais faire."]
ACCUSES = ["Tant mieux !", "Content de l'entendre !", "Parfait.", "Ravi de l'apprendre."]

with open(chemins.DIALOGUES, 'r', encoding='utf-8') as f:
    QUESTIONS_REELLES = [(d['q'], d['a']) for d in map(json.loads, f)]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# On cherche les erreurs du modèle de l'étape 2, c'est lui que le DPO va corriger
fichier_modele = chemins.MODELE_CHAT
modele = MonLLM.charger(fichier_modele, device)
taille_contexte = modele.taille_contexte
tok = decoupage(modele.embed_tokens.num_embeddings)

@torch.no_grad()
def repondre(contexte):
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

def en_boucle(texte):
    if re.search(r'(.)\1{6,}', texte):
        return True
    mots = texte.lower().split()
    return len(mots) >= 6 and len(set(mots)) <= len(mots) / 2.5

def conversation(faits_donnes, valeurs):
    return ''.join(
        f"{UTILISATEUR}{FAITS[nom]['dire'].format(v=valeurs[nom])}\n"
        f"{ASSISTANT}{FAITS[nom]['accuse'].format(v=valeurs[nom])}\n"
        for nom in faits_donnes)

paires = []
corrections = 0

def ajouter(contexte, choisie, rejetee, est_correction):
    global corrections
    if choisie.strip() and choisie.strip() != rejetee.strip():
        paires.append({'contexte': contexte, 'choisie': choisie, 'rejetee': rejetee,
                       'origine': 'auto'})
        corrections += est_correction

# L'âge sert seulement à apprendre à dire « je sais pas », les autres faits sont toujours donnés avant
FAIT_INCONNU = 'âge'
FAITS_CONNUS = [n for n in FAITS if n != FAIT_INCONNU]

for numero in range(NB_SCENARIOS):
    categorie = numero % 10

    if categorie < 6:
        cible = FAITS_CONNUS[categorie % len(FAITS_CONNUS)]
        voisins = [n for n in FAITS_CONNUS if n != cible]
        donnes = [cible, random.choice(voisins)]
        random.shuffle(donnes)
        valeurs = {n: FAITS[n]['valeur']() for n in donnes}

        contexte = conversation(donnes, valeurs) + UTILISATEUR + FAITS[cible]['question'] + '\n' + ASSISTANT
        ideale = FAITS[cible]['reponse'].format(v=valeurs[cible])
        reponse = repondre(contexte)
        faux = valeurs[cible].lower() not in reponse.lower()
        voisin = next(n for n in donnes if n != cible)
        fabriquee = FAITS[voisin]['reponse'].format(v=valeurs[voisin])
        # S'il a juste on préfère sa propre réponse, et on fabrique quand même une mauvaise :
        # sans ces paires il désapprend ce qu'il savait déjà faire
        ajouter(contexte, ideale if faux else reponse, reponse if faux else fabriquee, faux)

    elif categorie < 8:
        donnes = random.sample(FAITS_CONNUS, 2)
        valeurs = {n: FAITS[n]['valeur']() for n in donnes}

        contexte = (conversation(donnes, valeurs)
                    + UTILISATEUR + FAITS[FAIT_INCONNU]['question'] + '\n' + ASSISTANT)
        ideale = random.choice(REFUS)
        reponse = repondre(contexte)
        faux = not ('pas' in reponse.lower() and 'dit' in reponse.lower())
        invention = FAITS[FAIT_INCONNU]['reponse'].format(v=FAITS[FAIT_INCONNU]['valeur']())
        ajouter(contexte, ideale if faux else reponse, reponse if faux else invention, faux)

    elif categorie < 9:
        precedente = "Très bien, merci. Et vous ?"
        contexte = (f"{UTILISATEUR}comment ça va ?\n{ASSISTANT}{precedente}\n"
                    f"{UTILISATEUR}{random.choice(['ça va aussi', 'je vais bien', 'ça va merci', 'très bien aussi'])}\n{ASSISTANT}")
        reponse = repondre(contexte)
        faux = reponse.strip() == precedente or en_boucle(reponse)
        ajouter(contexte, random.choice(ACCUSES) if faux else reponse, reponse if faux else precedente, faux)

    else:
        question, ideale = random.choice(QUESTIONS_REELLES)
        contexte = UTILISATEUR + question + '\n' + ASSISTANT
        reponse = repondre(contexte)
        faux = en_boucle(reponse) or 'pas dit' in reponse or 'saurais pas' in reponse
        ajouter(contexte, ideale, reponse if faux else random.choice(REFUS + AVEUX), faux)

with open(FICHIER_PREFERENCES, 'a', encoding='utf-8') as f:
    for paire in paires:
        f.write(json.dumps(paire, ensure_ascii=False) + '\n')

print(f"{NB_SCENARIOS} situations testées -> {len(paires)} paires ajoutées à {FICHIER_PREFERENCES}")
print(f"  dont {corrections} erreurs réelles du modèle, {len(paires)-corrections} paires de protection")
