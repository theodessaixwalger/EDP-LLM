import datetime
import functools
import json
import random
import re
import zlib
import numpy as np
import torch
import chemins
from model import MonLLM
from outils import calcul, horloge
from recherche import Recherche, ligne_document
from tokeniseur import decoupage

random.seed(1337)

# --- Configuration ---
taille_contexte = 4096
nb_conversations_train = 40000
nb_conversations_val = 4000
longueur_max_replique = 100

UTILISATEUR = "Utilisateur : "
ASSISTANT = "Assistant : "

etat = torch.load(chemins.MODELE_BASE, map_location='cpu', weights_only=True)
tok = decoupage(MonLLM.architecture(etat)['taille_vocab'])
del etat
vocabulaire = tok.alphabet

# --- Répliques extraites des romans ---
with open(chemins.CORPUS_ROMANS, 'r', encoding='utf-8') as f:
    texte = f.read()
paragraphes = [re.sub(r'\s*\n+\s*', ' ', p).strip() for p in re.split(r'\n{3,}', texte)]

# Dans les romans une réplique commence par un tiret, deux répliques qui se suivent font un échange
def replique(paragraphe):
    if paragraphe.startswith('--') or paragraphe.startswith('—'):
        return paragraphe.lstrip('-— ')
    return None

paires = []
for a, b in zip(paragraphes, paragraphes[1:]):
    question, reponse = replique(a), replique(b)
    if question and reponse and len(question) <= longueur_max_replique and len(reponse) <= longueur_max_replique \
            and set(question + reponse) <= vocabulaire:
        paires.append((question, reponse))

random.shuffle(paires)
coupure = int(len(paires) * 0.9)
paires_train, paires_val = paires[:coupure], paires[coupure:]
print(f"Échanges extraits des romans : {len(paires):,}")

# --- Vraies questions-réponses (dialogues.py) ---
with open(chemins.DIALOGUES, 'r', encoding='utf-8') as f:
    dialogues = [json.loads(ligne) for ligne in f]
    dialogues = [(d['q'], d['a']) for d in dialogues]
random.shuffle(dialogues)
coupure = int(len(dialogues) * 0.9)
dialogues_train, dialogues_val = dialogues[:coupure], dialogues[coupure:]
print(f"Échanges assistant réels : {len(dialogues):,}")

# --- Faits personnels ---
PRENOMS = ["Théo", "Julie", "Lucas", "Emma", "Hugo", "Léa", "Nathan", "Chloé", "Louis", "Manon",
           "Jules", "Camille", "Arthur", "Inès", "Paul", "Sarah", "Adam", "Zoé", "Gabriel", "Alice",
           "Maxime", "Clara", "Victor", "Jade"]
VILLES = ["Paris", "Lyon", "Marseille", "Toulouse", "Nantes", "Bordeaux", "Lille", "Rennes",
          "Strasbourg", "Montpellier", "Nice", "Brest", "Grenoble", "Dijon", "Angers", "Rouen", "Tours", "Metz"]
METIERS = ["boulanger", "infirmier", "développeur", "professeur", "architecte", "plombier", "journaliste",
           "pilote", "cuisinier", "médecin", "musicien", "photographe", "avocat", "jardinier"]
ANIMAUX = ["chien", "chat", "lapin", "cheval", "hamster", "perroquet", "poisson"]
COULEURS = ["rouge", "bleu", "vert", "jaune", "violet", "noir", "blanc", "rose", "gris", "marron"]

# La plupart du temps on invente le nom : impossible à apprendre par cœur, il doit vraiment le recopier
def nom_invente():
    nom = ''.join(random.choice('aeiouyéèô' if random.random() < 0.4 else 'bcdfghjklmnpqrstvwxz')
                  for _ in range(random.randint(2, 9))).capitalize()
    if random.random() < 0.1:
        nom += '-' + nom_invente()
    return nom

def choisir(liste):
    return random.choice(liste) if random.random() < 0.3 else nom_invente()

FAITS = {
    'prenom': {
        'valeur': lambda: {'v': choisir(PRENOMS)},
        'dire': ["Je m'appelle {v}.", "Mon prénom est {v}.", "Moi, c'est {v}."],
        'accuser': ["Enchanté, {v} !", "Ravi de vous rencontrer, {v}.", "C'est noté."],
        'demander': ["Comment je m'appelle ?", "Quel est mon prénom ?", "Tu te souviens de mon prénom ?"],
        'repondre': ["Vous vous appelez {v}.", "Votre prénom est {v}."],
    },
    'ville': {
        'valeur': lambda: {'v': choisir(VILLES)},
        'dire': ["J'habite à {v}.", "Je vis à {v}."],
        'accuser': ["C'est une belle ville, {v}.", "C'est noté, vous habitez à {v}.", "D'accord."],
        'demander': ["Où est-ce que j'habite ?", "Dans quelle ville je vis ?", "Tu sais où j'habite ?"],
        'repondre': ["Vous habitez à {v}.", "Vous vivez à {v}."],
    },
    'age': {
        'valeur': lambda: {'v': str(random.randint(8, 95))},
        'dire': ["J'ai {v} ans.", "Je viens d'avoir {v} ans."],
        'accuser': ["C'est noté, {v} ans.", "Très bien."],
        'demander': ["Quel âge j'ai ?", "Tu te rappelles mon âge ?"],
        'repondre': ["Vous avez {v} ans."],
    },
    'metier': {
        'valeur': lambda: {'v': random.choice(METIERS)},
        'dire': ["Je travaille comme {v}.", "Mon métier : {v}."],
        'accuser': ["Le métier de {v}, c'est passionnant.", "Intéressant !"],
        'demander': ["Quel est mon métier ?", "Qu'est-ce que je fais comme travail ?"],
        'repondre': ["Vous travaillez comme {v}.", "Votre métier est {v}."],
    },
    'animal': {
        'valeur': lambda: {'a': random.choice(ANIMAUX), 'v': choisir(PRENOMS)},
        'dire': ["J'ai un {a} qui s'appelle {v}.", "Mon {a} s'appelle {v}."],
        'accuser': ["C'est un joli nom pour un {a}, {v}.", "C'est noté."],
        'demander': ["Comment s'appelle mon {a} ?", "Tu te souviens du nom de mon {a} ?"],
        'repondre': ["Votre {a} s'appelle {v}.", "Il s'appelle {v}."],
    },
    'couleur': {
        'valeur': lambda: {'v': random.choice(COULEURS)},
        'dire': ["Ma couleur préférée est le {v}.", "J'adore le {v}, c'est ma couleur préférée."],
        'accuser': ["Le {v}, bon choix !", "C'est noté."],
        'demander': ["Quelle est ma couleur préférée ?", "Tu sais quelle couleur je préfère ?"],
        'repondre': ["Votre couleur préférée est le {v}.", "Vous préférez le {v}."],
    },
}

INCONNU = ["Vous ne me l'avez pas encore dit.", "Je ne sais pas, vous ne me l'avez pas dit."]

BAVARDAGE = [
    ("Bonjour !", "Bonjour ! Que puis-je faire pour vous ?"),
    ("Salut.", "Salut ! Comment allez-vous ?"),
    ("Comment ça va ?", "Très bien, merci. Et vous ?"),
    ("Ça va bien, merci.", "Tant mieux !"),
    ("Bien, et vous ?", "Très bien, merci !"),
    ("Merci beaucoup.", "Avec plaisir."),
    ("D'accord.", "Parfait."),
    ("Au revoir.", "Au revoir, à bientôt !"),
    ("Qui es-tu ?", "Je suis un petit modèle de langage, entraîné sur des romans et sur Wikipédia."),
    ("Qu'est-ce que tu sais faire ?", "Je peux discuter avec vous, retenir ce que vous me dites et chercher des réponses dans Wikipédia."),
    ("Tu as une mémoire ?", "Oui, je me souviens de ce que vous me dites pendant notre conversation."),
]

# --- Lecture de documents ---
# 35 % des conversations finissent par une question avec documents. Des fois le passage ne répond pas
# et il doit le dire, et souvent on ajoute un 2e passage pas bon (un leurre) comme en vrai
part_documents = 0.35
part_sans_reponse = 0.15
part_definitions = 0.40
part_leurres = 0.7
AUCUNE_REPONSE = ["Je ne trouve pas la réponse dans ce document.",
                  "Ce document ne répond pas à la question."]

def dans_le_vocabulaire(texte):
    return tok.nettoyer(texte)[0]

def est_val(texte):
    return zlib.crc32(texte.encode()) % 20 == 0

questions_documents = {'train': {'avec': [], 'sans': []}, 'val': {'avec': [], 'sans': []}}
with open(chemins.QUESTIONS_DOCUMENTS, 'r', encoding='utf-8') as f:
    for ligne in f:
        q = json.loads(ligne)
        if set(q['question'] + q['reponse']) - vocabulaire:
            continue
        cote = 'val' if est_val(q['passage']) else 'train'
        questions_documents[cote]['avec' if q['reponse'] else 'sans'].append(
            (q['question'], q['reponse'], dans_le_vocabulaire(q['passage'])))
print(f"Questions sur documents : {len(questions_documents['train']['avec']):,} avec réponse, "
      f"{len(questions_documents['train']['sans']):,} sans")

DEFINIR = ["C'est quoi {s} ?", "Qu'est-ce que {s} ?", "Parle-moi de {s}.", "Que sais-tu sur {s} ?",
           "Explique-moi {s}.", "Tu connais {s} ?", "Dis-moi ce que tu sais sur {s}.", "{s}, c'est quoi ?"]
PRESENTER = ["Qui est {s} ?", "Qui était {s} ?", "Parle-moi de {s}.", "Que sais-tu sur {s} ?",
             "Tu connais {s} ?"]

# Fabrique « C'est quoi X ? » à partir du début d'un article, la réponse c'est ses premières phrases
def question_de_definition(titre, passage):
    sujet = re.sub(r'\s*\(.*\)', '', titre)
    article = re.match(r"(Les |Le |La |L')" + re.escape(sujet), passage, re.IGNORECASE)
    if article:
        sujet = article.group(0)[0].lower() + article.group(0)[1:]
    modeles = PRESENTER if re.search(r'\bnée? (le|en|à)\b', passage[:250]) else DEFINIR
    question = random.choice(modeles).format(s=sujet)
    question = question[0].upper() + question[1:]
    reponse = ''
    for phrase in re.split(r'(?<=[.!?]) ', passage):
        if len(reponse) >= 60 or len(reponse) + len(phrase) > 300:
            break
        reponse = (reponse + ' ' + phrase).strip()
    return question, reponse

recherche = Recherche()
definitions = {'train': [], 'val': []}
for numero in random.sample(range(recherche.nb_passages), min(60000, recherche.nb_passages)):
    titre, passage = recherche.passage(numero)
    passage = dans_le_vocabulaire(passage)
    question, reponse = question_de_definition(dans_le_vocabulaire(titre), passage)
    if len(reponse) >= 40 and reponse in passage:
        definitions['val' if est_val(passage) else 'train'].append((question, reponse, passage))
print(f"Questions de définition : {len(definitions['train']):,}")

def leurre(question, passage):
    for _, numero in recherche.chercher(question, n=2):
        autre = dans_le_vocabulaire(recherche.passage(numero)[1])
        if autre != passage:
            return autre
    return None

# --- Outils ---
part_outils = 0.2
EN_LETTRES = {0: 'zéro', 1: 'un', 2: 'deux', 3: 'trois', 4: 'quatre', 5: 'cinq', 6: 'six', 7: 'sept',
              8: 'huit', 9: 'neuf', 10: 'dix', 11: 'onze', 12: 'douze', 15: 'quinze', 20: 'vingt',
              30: 'trente', 40: 'quarante', 50: 'cinquante', 60: 'soixante', 100: 'cent', 1000: 'mille'}
OPERATIONS = {'+': ['plus', '+'], '-': ['moins', '-'], '*': ['fois', 'x', '*', '×', 'multiplié par'],
              '/': ['divisé par', '/']}
DEMANDES_DE_CALCUL = ["Combien font {} ?", "combien fait {}", "{}", "{} = ?", "Calcule {}.",
                      "Quel est le résultat de {} ?", "Tu peux calculer {} ?", "ça fait combien {} ?"]
DEMANDES_DE_DATE = ["Quel jour sommes-nous ?", "on est quel jour", "Quelle est la date d'aujourd'hui ?",
                    "c'est quoi la date d'aujourd'hui", "on est le combien aujourd'hui ?",
                    "En quelle année sommes-nous ?", "Quelle heure est-il ?", "il est quelle heure"]

def nombre_au_hasard():
    n = random.choice([random.randint(0, 12), random.randint(0, 100), random.randint(0, 10000)])
    return EN_LETTRES[n] if n in EN_LETTRES and random.random() < 0.4 else str(n)

# Le résultat de la calculatrice ou de l'horloge est mis en document, il doit juste le recopier
def echange_outil():
    while True:
        if random.random() < 0.25:
            question = random.choice(DEMANDES_DE_DATE)
            instant = datetime.datetime(random.randint(2020, 2035), random.randint(1, 12), random.randint(1, 28),
                                        random.randint(0, 23), random.randint(0, 59))
            resultat = horloge(question, instant)
        else:
            if random.random() < 0.1:
                operation = f"{random.randint(1, 99)} % de {random.randint(1, 1000)}"
            else:
                symbole = random.choice(list(OPERATIONS))
                operation = f"{nombre_au_hasard()} {random.choice(OPERATIONS[symbole])} {nombre_au_hasard()}"
            question = random.choice(DEMANDES_DE_CALCUL).format(operation)
            resultat = calcul(question)
        if resultat:
            return question, resultat, ligne_document(resultat)

def echange_document(cote):
    if random.random() < part_outils:
        question, reponse, documents = echange_outil()
        return (taper(question, random.random() < 0.3), reponse, False, documents)
    tirage = random.random()
    if tirage < part_sans_reponse:
        question, _, passage = random.choice(questions_documents[cote]['sans'])
        reponse = random.choice(AUCUNE_REPONSE)
        passages = [passage]
    else:
        source = definitions[cote] if tirage < part_sans_reponse + part_definitions else questions_documents[cote]['avec']
        question, reponse, passage = random.choice(source)
        passages = [passage]
        if random.random() < part_leurres:
            autre = leurre(question, passage)
            if autre:
                passages.insert(random.randint(0, 1), autre)
    documents = ''.join(ligne_document(p) for p in passages)
    return (taper(question, random.random() < 0.3), reponse, False, documents)

# --- Façons de taper ---
# Variantes de frappe pour qu'il reconnaisse « je m'appelle léo » autant que « Je m'appelle Léo. »
def taper(phrase, minuscules):
    if minuscules:
        phrase = phrase.lower()
    tirage = random.random()
    if tirage < 0.4:
        phrase = phrase.rstrip(' .!?')
    elif tirage < 0.55:
        phrase = phrase.replace(' ?', '?').replace(' !', '!')
    return phrase

# --- Construction des conversations ---
def taille_echange(echange):
    return _taille(echange[0], echange[1], ''.join(echange[3:]))

@functools.lru_cache(maxsize=None)
def _taille(question, reponse, documents):
    return len(tok.encoder(documents + UTILISATEUR + question + '\n' + ASSISTANT)) + len(tok.encoder(reponse + '\n'))

def conversation(paires, dialogues, reserve=0):
    connus = {}
    echanges = []
    longueur = 0
    budget = random.randint(90, taille_contexte + 1 - reserve)
    for _ in range(400):
        tirage = random.random()
        memoire = False

        if tirage < 0.28:
            nom = random.choice(list(FAITS))
            valeur = FAITS[nom]['valeur']()
            minuscules = random.random() < 0.4
            if minuscules:
                valeur = {k: v.lower() for k, v in valeur.items()}
            phrase = random.choice(FAITS[nom]['dire']).format(**valeur)
            # Un fait redonné remplace l'ancien
            if nom in connus:
                phrase = "En fait, " + phrase[0].lower() + phrase[1:]
            connus[nom] = valeur
            echange = (taper(phrase, minuscules), random.choice(FAITS[nom]['accuser']).format(**valeur))

        elif tirage < 0.56:
            memoire = True
            inconnus = [n for n in FAITS if n not in connus]
            if connus and (not inconnus or random.random() < 0.75):
                nom = random.choice(list(connus))
                valeur = connus[nom]
                reponse = random.choice(FAITS[nom]['repondre']).format(**valeur)
            else:
                nom = random.choice(inconnus)
                valeur = FAITS[nom]['valeur']()
                reponse = random.choice(INCONNU)
            question = random.choice(FAITS[nom]['demander']).format(**valeur)
            echange = (taper(question, random.random() < 0.4), reponse)

        elif tirage < 0.61:
            question, reponse = random.choice(BAVARDAGE)
            echange = (taper(question, random.random() < 0.4), reponse)
        elif tirage < 0.93:
            echange = random.choice(dialogues)
        else:
            echange = random.choice(paires)

        taille = taille_echange(echange)
        if echanges and longueur + taille > budget:
            break
        echanges.append(echange + (memoire,))
        longueur += taille
    return echanges

# --- Conversations à longue portée ---
# Faits donnés au début, jamais répétés, redemandés à la fin : sans ça il oubliait au-delà de ~2000 caractères
part_longue_portee = 0.4

def conversation_longue_portee(paires, dialogues, reserve=0):
    budget = random.randint(500, taille_contexte + 1 - reserve)
    donnes = random.sample(list(FAITS), random.randint(1, 3))
    debut, fin = [], []
    for nom in donnes:
        valeur = FAITS[nom]['valeur']()
        minuscules = random.random() < 0.4
        if minuscules:
            valeur = {k: v.lower() for k, v in valeur.items()}
        phrase = random.choice(FAITS[nom]['dire']).format(**valeur)
        debut.append((taper(phrase, minuscules), random.choice(FAITS[nom]['accuser']).format(**valeur), False))
        question = random.choice(FAITS[nom]['demander']).format(**valeur)
        fin.append((taper(question, random.random() < 0.4),
                    random.choice(FAITS[nom]['repondre']).format(**valeur), True))
    random.shuffle(fin)

    longueur = sum(map(taille_echange, debut + fin))
    jamais_donnes = [n for n in FAITS if n not in donnes]
    milieu = []
    for _ in range(400):
        tirage = random.random()
        if tirage < 0.15:
            nom = random.choice(jamais_donnes)
            question = random.choice(FAITS[nom]['demander']).format(**FAITS[nom]['valeur']())
            echange = (taper(question, random.random() < 0.4), random.choice(INCONNU), True)
        elif tirage < 0.25:
            question, reponse = random.choice(BAVARDAGE)
            echange = (taper(question, random.random() < 0.4), reponse, False)
        elif tirage < 0.75:
            echange = random.choice(dialogues) + (False,)
        else:
            echange = random.choice(paires) + (False,)
        if longueur + taille_echange(echange) > budget:
            break
        milieu.append(echange)
        longueur += taille_echange(echange)
    return debut + milieu + fin

def tirer_conversation(paires, dialogues, cote):
    fin = [echange_document(cote)] if random.random() < part_documents else []
    reserve = sum(map(taille_echange, fin))
    if random.random() < part_longue_portee:
        return conversation_longue_portee(paires, dialogues, reserve) + fin
    return conversation(paires, dialogues, reserve) + fin

# --- Compréhension de lecture (lecture.py) ---
# On peut ajouter des conversations qui finissent par un exemple de lecture (part_lecture du total),
# sans toucher aux 40 000 autres. Elles ont la même part de longue portée que les autres.
# Désactivé : à 25 %, le modèle apprend le format mais pas à choisir la bonne réponse dans le document,
# il recopie surtout le premier choix de la liste. QCM v2 avec documents, génération au vote :
# 45 % -> 24 % avec « B) Lyon. », 27 % avec « Lyon. ». Les modèles sont dans modeles/60M_lecture_*
part_lecture = 0
lectures = {'train': [], 'val': []}
if part_lecture:
    with open(chemins.LECTURE, 'r', encoding='utf-8') as f:
        for ligne in f:
            exemple = json.loads(ligne)
            lectures['val' if est_val(exemple['article']) else 'train'].append(exemple)

def echange_lecture(exemple):
    question = dans_le_vocabulaire(exemple['question'])
    # Le QCM garde exactement le format de evaluer_qcm.py, le reste est tapé comme un vrai message
    if 'choix' not in exemple:
        question = taper(question, random.random() < 0.3)
    reponse = dans_le_vocabulaire(exemple['reponse'])
    if not exemple['documents']:
        return (question, reponse, False)
    return (question, reponse, False, ''.join(ligne_document(dans_le_vocabulaire(d)) for d in exemple['documents']))

def conversation_lecture(exemple, paires, dialogues):
    fin = [echange_lecture(exemple)]
    reserve = sum(map(taille_echange, fin))
    if random.random() < part_longue_portee:
        return conversation_longue_portee(paires, dialogues, reserve) + fin
    return conversation(paires, dialogues, reserve) + fin

def tirer_lectures(cote, nombre):
    tas = lectures[cote][:]
    random.shuffle(tas)
    # Pas assez d'exemples (côté validation) : on en reprend au hasard
    return tas[:nombre] if len(tas) >= nombre else tas + random.choices(tas, k=nombre - len(tas))

# --- Encodage ---
def encoder(conversations, avec_mesures):
    ids = np.full((len(conversations), taille_contexte + 1), tok.saut_de_ligne, dtype=np.uint16)
    masques = {nom: np.zeros((len(conversations), taille_contexte + 1), dtype=bool)
               for nom in (('masque', 'memoire', 'document') if avec_mesures else ('masque',))}
    for n, echanges in enumerate(conversations):
        codes, a_apprendre, a_retrouver, a_lire = [], [], [], []
        for question, reponse, est_memoire, *documents in echanges:
            entree = tok.encoder(''.join(documents) + UTILISATEUR + question + '\n' + ASSISTANT)
            sortie = tok.encoder(reponse + '\n')
            codes += entree + sortie
            # masque : la perte se calcule que sur les réponses. memoire et document servent à choisir
            # le meilleur modèle dans finetune.py
            a_apprendre += [False] * len(entree) + [True] * len(sortie)
            a_retrouver += [False] * len(entree) + [est_memoire] * len(sortie)
            a_lire += [False] * len(entree) + [bool(documents)] * len(sortie)
        codes = codes[:taille_contexte + 1]
        ids[n, :len(codes)] = codes
        for nom, valeurs in (('masque', a_apprendre), ('memoire', a_retrouver), ('document', a_lire)):
            if nom in masques:
                masques[nom][n, :len(codes)] = valeurs[:len(codes)]
    return ids, masques

conversations_train = [tirer_conversation(paires_train, dialogues_train, 'train') for _ in range(nb_conversations_train)]
conversations_val = [tirer_conversation(paires_val, dialogues_val, 'val') for _ in range(nb_conversations_val)]

if part_lecture:
    nb_lecture_train = round(nb_conversations_train * part_lecture / (1 - part_lecture))
    nb_lecture_val = round(nb_conversations_val * part_lecture / (1 - part_lecture))
    print(f"Exemples de lecture : {len(lectures['train']):,} pour l'entraînement, {len(lectures['val']):,} pour la validation")
    conversations_train += [conversation_lecture(e, paires_train, dialogues_train) for e in tirer_lectures('train', nb_lecture_train)]
    conversations_val += [conversation_lecture(e, paires_val, dialogues_val) for e in tirer_lectures('val', nb_lecture_val)]

FICHIERS = {'train': chemins.CHAT_TRAIN, 'val': chemins.CHAT_VAL}

for nom, conversations in (('train', conversations_train), ('val', conversations_val)):
    ids, masques = encoder(conversations, avec_mesures=(nom == 'val'))
    np.savez(FICHIERS[nom], ids=ids, **masques)
    moyenne = sum(len(c) for c in conversations) / len(conversations)
    avec_documents = sum(len(c[-1]) == 4 for c in conversations) / len(conversations)
    print(f"{nom:5} : {len(conversations):,} conversations, {moyenne:.1f} échanges en moyenne, "
          f"{avec_documents:.0%} finissent par une question sur document")

print("\nExemple de conversation :\n")
exemple = next(c for c in conversations_train if len(c) >= 5 and len(c[-1]) == 4)
for question, reponse, _, *documents in exemple[-4:]:
    print(''.join(documents) + UTILISATEUR + question)
    print(ASSISTANT + reponse)

if part_lecture:
    print("\nFin d'une conversation de lecture :\n")
    for question, reponse, _, *documents in conversations_train[nb_conversations_train + 1][-2:]:
        print(''.join(documents) + UTILISATEUR + question)
        print(ASSISTANT + reponse)

print(f"\nTerminé : {chemins.CHAT_TRAIN}, {chemins.CHAT_VAL} créés")
