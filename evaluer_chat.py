import os
import random
import re
import sys
import torch
import torch.nn.functional as F
from model import MonLLM
from outils import utiliser as utiliser_outil
from recherche import Recherche, ligne_document
from tokeniseur import decoupage
import chemins

# python evaluer_chat.py [modele.pt] : rejoue toujours les mêmes situations et compte ce qui marche
FICHIER = sys.argv[1] if len(sys.argv) > 1 else chemins.MODELE_CHAT
NB_TESTS = 40
temperature = 0.5
longueur_max_reponse = 150

UTILISATEUR = "Utilisateur : "
ASSISTANT = "Assistant : "
PRENOMS = ["Théo", "Julie", "Lucas", "Emma", "Hugo", "Léa", "Nathan", "Chloé"]
VILLES = ["Brest", "Lyon", "Nantes", "Bordeaux", "Lille", "Rennes", "Dijon"]
QUESTIONS_OUVERTES = ["raconte moi une blague", "c'est quoi la photosynthèse ?",
                      "tu peux m'aider en maths ?", "explique moi la gravité",
                      "qu'est-ce que tu penses de la pluie ?"]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

modele = MonLLM.charger(FICHIER, device)
taille_contexte = modele.taille_contexte
tok = decoupage(modele.embed_tokens.num_embeddings)
unite = 'caractères' if len(tok.morceaux) == len(tok.alphabet) else 'morceaux'

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

# --- Culture générale ---
# Le modèle note les 4 réponses, on regarde si la bonne (toujours la 1re) arrive en tête. Le hasard fait 25 %
QCM = [
    ("Quelle est la capitale de la France ?", ["Paris", "Lyon", "Marseille", "Bordeaux"]),
    ("Quelle est la capitale de l'Italie ?", ["Rome", "Milan", "Naples", "Venise"]),
    ("Quelle est la capitale de l'Allemagne ?", ["Berlin", "Munich", "Hambourg", "Francfort"]),
    ("Quelle est la capitale de l'Espagne ?", ["Madrid", "Barcelone", "Séville", "Valence"]),
    ("Quelle est la capitale du Royaume-Uni ?", ["Londres", "Manchester", "Liverpool", "Édimbourg"]),
    ("Quelle est la capitale du Japon ?", ["Tokyo", "Kyoto", "Osaka", "Pékin"]),
    ("Quel fleuve traverse Paris ?", ["La Seine", "La Loire", "Le Rhône", "La Garonne"]),
    ("Quel est le plus haut sommet des Alpes ?", ["Le mont Blanc", "Le Cervin", "Le mont Ventoux", "Le puy de Dôme"]),
    ("Dans quel pays se trouve la tour Eiffel ?", ["En France", "En Italie", "En Espagne", "En Belgique"]),
    ("Quelle langue parle-t-on au Brésil ?", ["Le portugais", "L'espagnol", "Le français", "L'anglais"]),
    ("Quel est le plus grand océan du monde ?", ["Le Pacifique", "L'Atlantique", "L'océan Indien", "L'Arctique"]),
    ("Quelle monnaie utilise-t-on en France ?", ["L'euro", "Le dollar", "Le yen", "La livre sterling"]),
    ("Qui a écrit Les Misérables ?", ["Victor Hugo", "Émile Zola", "Alexandre Dumas", "Gustave Flaubert"]),
    ("Qui a écrit Les Trois Mousquetaires ?", ["Alexandre Dumas", "Victor Hugo", "Jules Verne", "Stendhal"]),
    ("Qui a écrit Vingt mille lieues sous les mers ?", ["Jules Verne", "Victor Hugo", "Émile Zola", "Marcel Proust"]),
    ("Qui a écrit Madame Bovary ?", ["Gustave Flaubert", "Stendhal", "Victor Hugo", "Jules Verne"]),
    ("Qui a écrit Germinal ?", ["Émile Zola", "Victor Hugo", "Honoré de Balzac", "Alexandre Dumas"]),
    ("Qui a écrit Le Comte de Monte-Cristo ?", ["Alexandre Dumas", "Victor Hugo", "Jules Verne", "Émile Zola"]),
    ("Qui a peint la Joconde ?", ["Léonard de Vinci", "Pablo Picasso", "Vincent van Gogh", "Claude Monet"]),
    ("Qui fut le premier empereur des Français ?", ["Napoléon Bonaparte", "Louis XIV", "Charlemagne", "Jules César"]),
    ("En quelle année a eu lieu la prise de la Bastille ?", ["1789", "1815", "1492", "1914"]),
    ("Quelle planète est la plus proche du Soleil ?", ["Mercure", "Mars", "Jupiter", "Saturne"]),
    ("Quel est le satellite naturel de la Terre ?", ["La Lune", "Le Soleil", "Mars", "Vénus"]),
    ("Quelle est la formule chimique de l'eau ?", ["H2O", "CO2", "O2", "NaCl"]),
    ("Quel gaz respirons-nous pour vivre ?", ["L'oxygène", "L'hélium", "Le méthane", "L'azote liquide"]),
    ("À quelle température l'eau bout-elle ?", ["100 degrés", "50 degrés", "0 degré", "200 degrés"]),
    ("Quel organe pompe le sang dans le corps ?", ["Le cœur", "Le foie", "Le poumon", "L'estomac"]),
    ("Quel métal est attiré par un aimant ?", ["Le fer", "L'or", "L'argent", "Le cuivre"]),
    ("Que fabriquent les abeilles ?", ["Du miel", "Du lait", "De la soie", "Du beurre"]),
    ("Combien de pattes a une araignée ?", ["Huit", "Six", "Quatre", "Dix"]),
    ("Quel animal miaule ?", ["Le chat", "Le chien", "La vache", "Le cheval"]),
    ("Combien font deux plus deux ?", ["Quatre", "Trois", "Cinq", "Six"]),
    ("Combien font dix moins trois ?", ["Sept", "Six", "Huit", "Treize"]),
    ("Combien de côtés a un triangle ?", ["Trois", "Quatre", "Cinq", "Six"]),
    ("Combien de jours y a-t-il dans une semaine ?", ["Sept", "Cinq", "Dix", "Douze"]),
    ("Combien d'heures y a-t-il dans une journée ?", ["Vingt-quatre", "Douze", "Trente", "Soixante"]),
    ("En quelle saison fait-il le plus froid ?", ["En hiver", "En été", "Au printemps", "En automne"]),
    ("Quelle est la couleur du ciel par beau temps ?", ["Bleu", "Vert", "Rouge", "Noir"]),
    ("Quel est le contraire de chaud ?", ["Froid", "Grand", "Rapide", "Doux"]),
    ("Quel instrument a des touches noires et blanches ?", ["Le piano", "La guitare", "La flûte", "Le tambour"]),
]

@torch.no_grad()
def note(question, reponse):
    contexte = tok.encoder(UTILISATEUR + question + '\n' + ASSISTANT)
    ids = contexte + tok.encoder(reponse + '\n')
    logits, _ = modele(torch.tensor([ids[:-1]], device=device))
    log_probas = torch.log_softmax(logits[0], dim=-1)
    cibles = torch.tensor(ids[1:], device=device)
    detail = log_probas[torch.arange(len(cibles), device=device), cibles]
    return detail[len(contexte) - 1:].sum().item() / len(reponse + '\n')

def qcm():
    justes, erreurs = 0, []
    for question, reponses in QCM:
        notes = [note(question, r) for r in reponses]
        if max(range(len(notes)), key=notes.__getitem__) == 0:
            justes += 1
        else:
            erreurs.append(question)
    return justes / len(QCM), erreurs

# --- Culture générale avec documents ---
recherche = Recherche() if os.path.exists(os.path.join(chemins.RECHERCHE, 'positions.npy')) else None

def mot_cle(reponse):
    return re.sub(r"^(le |la |les |l'|en |du |de la |des |un |une )", '', reponse.lower())

def culture_avec_documents():
    lue, sans, apportee, exemples = 0, 0, 0, []
    for question, reponses in QCM:
        attendu = mot_cle(reponses[0])
        trouves = recherche.documents_pour(question)
        documents = ''.join(ligne_document(tok.nettoyer(p)[0]) for _, p in trouves)
        apportee += attendu in documents.lower()
        avec_documents = repondre(documents + UTILISATEUR + question + '\n' + ASSISTANT)
        lue += attendu in avec_documents.lower()
        sans += attendu in repondre(UTILISATEUR + question + '\n' + ASSISTANT).lower()
        exemples.append((question, [t for t, _ in trouves], avec_documents))
    n = len(QCM)
    return lue / n, sans / n, apportee / n, exemples

# --- Outils ---
DEMANDES_DE_CALCUL = ["Combien font {a} plus {b} ?", "combien fait {a} fois {b}", "{a} - {b}",
                      "Quel est le résultat de {a} divisé par {b} ?", "calcule {a} * {b}"]

def outils_recopies():
    justes, exemples = 0, []
    for _ in range(NB_TESTS):
        if random.random() < 0.25:
            question = random.choice(["Quel jour sommes-nous ?", "Quelle heure est-il ?"])
        else:
            question = random.choice(DEMANDES_DE_CALCUL).format(a=random.randint(2, 999), b=random.randint(2, 99))
        _, resultat = utiliser_outil(question)
        attendu = (resultat.split('= ')[1].rstrip('.') if '=' in resultat
                   else re.search(r"\d+ h \d+|\d+ \w+ \d{4}", resultat).group())
        reponse = repondre(ligne_document(resultat) + UTILISATEUR + question + '\n' + ASSISTANT)
        justes += attendu in reponse
        exemples.append((question, resultat, reponse))
    return justes / NB_TESTS, exemples

def en_boucle(texte):
    if re.search(r'(.)\1{6,}', texte):
        return True
    mots = texte.lower().split()
    return len(mots) >= 6 and len(set(mots)) <= len(mots) / 2.5

def deux_faits(prenom, ville):
    return (f"{UTILISATEUR}je m'appelle {prenom}\n{ASSISTANT}Enchanté, {prenom} !\n"
            f"{UTILISATEUR}j'habite à {ville}\n{ASSISTANT}C'est noté, vous habitez à {ville}.\n")

BAVARDAGE = [
    ("comment ça va ?", "Très bien, merci. Et vous ?"),
    ("ça va aussi", "Tant mieux !"),
    ("il fait beau aujourd'hui", "Parfait."),
    ("merci beaucoup", "Avec plaisir."),
    ("tu aimes lire ?", "Je ne saurais pas vous répondre."),
    ("d'accord", "Parfait."),
    ("raconte moi une blague", "Je ne sais pas expliquer ça."),
    ("tu connais Paris ?", "Désolé, c'est au delà de ce que je sais faire."),
]

DISTANCES = (2, 5, 8)
DISTANCES_LONGUES = (20, 40, 55, 150)

def remplissage(nb):
    tires = random.sample(BAVARDAGE, nb) if nb <= len(BAVARDAGE) else random.choices(BAVARDAGE, k=nb)
    return ''.join(f"{UTILISATEUR}{q}\n{ASSISTANT}{r}\n" for q, r in tires)

def recall_a_distance(nb_echanges):
    justes, distances = 0, []
    for _ in range(NB_TESTS):
        prenom = random.choice(PRENOMS)
        debut = f"{UTILISATEUR}je m'appelle {prenom}\n{ASSISTANT}Enchanté, {prenom} !\n"
        milieu = remplissage(nb_echanges)
        fin = UTILISATEUR + "comment je m'appelle\n" + ASSISTANT
        contexte = debut + milieu + fin
        distances.append(len(tok.encoder(contexte)))
        justes += prenom.lower() in repondre(contexte).lower()
    return justes / NB_TESTS, sum(distances) / len(distances)

def mesurer():
    resultats = {}
    random.seed(0)
    torch.manual_seed(0)

    for nom, question in (('rappel du prénom', "comment je m'appelle"),
                          ('rappel de la ville', "où j'habite")):
        justes = 0
        for _ in range(NB_TESTS):
            prenom, ville = random.choice(PRENOMS), random.choice(VILLES)
            attendu, interdit = (prenom, ville) if 'prénom' in nom else (ville, prenom)
            reponse = repondre(deux_faits(prenom, ville) + UTILISATEUR + question + '\n' + ASSISTANT)
            justes += attendu.lower() in reponse.lower() and interdit.lower() not in reponse.lower()
        resultats[nom] = justes / NB_TESTS

    justes = 0
    for _ in range(NB_TESTS):
        prenom, ville = random.choice(PRENOMS), random.choice(VILLES)
        reponse = repondre(deux_faits(prenom, ville) + UTILISATEUR + "quel âge j'ai\n" + ASSISTANT)
        justes += 'pas' in reponse.lower() and 'dit' in reponse.lower()
    resultats['avoue ne pas savoir'] = justes / NB_TESTS

    justes = 0
    precedente = "Très bien, merci. Et vous ?"
    for _ in range(NB_TESTS):
        suite = random.choice(['ça va aussi', 'je vais bien', 'ça va merci', 'très bien aussi'])
        reponse = repondre(f"{UTILISATEUR}comment ça va ?\n{ASSISTANT}{precedente}\n"
                           f"{UTILISATEUR}{suite}\n{ASSISTANT}")
        justes += reponse.strip() != precedente and not en_boucle(reponse)
        resultats['_dernier_bavardage'] = reponse
    resultats['bavardage non répété'] = justes / NB_TESTS

    justes = 0
    for _ in range(NB_TESTS):
        reponse = repondre(UTILISATEUR + random.choice(QUESTIONS_OUVERTES) + '\n' + ASSISTANT)
        justes += not en_boucle(reponse) and len(reponse.strip()) > 3
        resultats['_derniere_ouverte'] = reponse
    resultats['question ouverte lisible'] = justes / NB_TESTS

    for nb in DISTANCES:
        score, longueur = recall_a_distance(nb)
        resultats[f'prénom après {nb} échanges'] = score
        resultats[f'_distance_{nb}'] = longueur

    for nb in DISTANCES_LONGUES:
        score, longueur = recall_a_distance(nb)
        resultats[f'_long_{nb}'] = score
        resultats[f'_distance_{nb}'] = longueur
    return resultats

resultats = mesurer()
mesures = {k: v for k, v in resultats.items() if not k.startswith('_')}
print(f"Modèle évalué : {FICHIER}  ({NB_TESTS} tirages par mesure)\n")
for nom, score in mesures.items():
    barre = '#' * round(score * 30)
    print(f"  {nom:26} {score:6.0%} {barre}")
print(f"\n  {'MOYENNE':26} {sum(mesures.values())/len(mesures):6.0%}")
print("\nMémoire longue (hors moyenne) :")
for nb in DISTANCES_LONGUES:
    score = resultats[f'_long_{nb}']
    print(f"  {f'prénom après {nb} échanges':26} {score:6.0%} {'#' * round(score * 30)}")
score_qcm, erreurs_qcm = qcm()
print(f"\nCulture générale (hors moyenne, QCM de {len(QCM)} questions, hasard = 25 %) :")
print(f"  {'bonnes réponses':26} {score_qcm:6.0%} {'#' * round(score_qcm * 30)}")
print(f"  ratées : {' / '.join(q[:38] for q in erreurs_qcm[:6])}{' ...' if len(erreurs_qcm) > 6 else ''}")
if recherche:
    torch.manual_seed(0)
    lue, sans, apportee, exemples = culture_avec_documents()
    print(f"\nCulture générale en discussion (hors moyenne, mêmes {len(QCM)} questions, réponse écrite) :")
    print(f"  {'avec Wikipédia':26} {lue:6.0%} {'#' * round(lue * 30)}")
    print(f"  {'sans Wikipédia':26} {sans:6.0%} {'#' * round(sans * 30)}")
    print(f"  {'réponse dans les documents':26} {apportee:6.0%}   (ce que la recherche lui apporte)")
    for question, titres, reponse in exemples[:3] + exemples[12:14]:
        print(f"    {question}  [{', '.join(titres) or 'aucun document'}]\n      -> {reponse[:110]}")
torch.manual_seed(0)
score_outils, exemples = outils_recopies()
print(f"\nOutils (hors moyenne, {NB_TESTS} calculs et dates, résultat placé en document) :")
print(f"  {'résultat recopié':26} {score_outils:6.0%} {'#' * round(score_outils * 30)}")
for question, resultat, reponse in exemples[:4]:
    print(f"    {question}  [{resultat}]\n      -> {reponse[:80]}")
print(f"\nFenêtre de {taille_contexte} {unite}. Longueur des contextes testés :")
for nb in DISTANCES + DISTANCES_LONGUES:
    longueur = resultats[f'_distance_{nb}']
    tient = 'tient' if longueur <= taille_contexte else f'tronqué de {longueur - taille_contexte:.0f}'
    print(f"  après {nb} échanges : {longueur:.0f} {unite} ({tient})")
print("\nExemples de réponses :")
print(f"  bavardage       : {resultats['_dernier_bavardage'][:100]}")
print(f"  question ouverte: {resultats['_derniere_ouverte'][:100]}")
