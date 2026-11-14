import glob
import json
import os
import re
import sys
import time

import chemins
from recherche import SEUIL, Recherche, ecrire_index, est_question_de_culture, normaliser, reparer

# python recherche_marketing.py                 -> construit l'index marketing (Wikipédia, web, fiches)
# python recherche_marketing.py une question    -> montre ce que chaque mode trouve

# --- Découpage en passages ---
# Tout le texte de chaque document, en passages de ~700 caractères coupés en fin de phrase. Chaque passage
# reprend les dernières phrases du précédent, pour qu'une idée coupée en deux se lise en entier quelque part
TAILLE_PASSAGE = 700
CHEVAUCHEMENT = 150
LONGUEUR_MINIMALE_PASSAGE = 100

FIN_DE_PHRASE = re.compile(r"(?<=[.!?…»])\s+(?=[«\"(A-ZÀ-ÖØ-Þ0-9])")
# Ce qui suit ces titres de section, dans Wikipédia, n'est que des listes de liens
FIN_D_ARTICLE = re.compile(r"^(Voir aussi|Notes et références|Références|Bibliographie|Liens externes|Articles connexes)\s*$",
                           re.M)

def phrases(texte):
    resultat = []
    for ligne in texte.split('\n'):
        ligne = ' '.join(ligne.split())
        if not ligne:
            continue
        # Un titre de section ou une ligne de liste sans ponctuation devient une phrase à part
        if ligne[-1] not in '.!?:;…»)':
            ligne += '.'
        for phrase in FIN_DE_PHRASE.split(ligne):
            while len(phrase) > TAILLE_PASSAGE:
                coupe = phrase.rfind(' ', 0, TAILLE_PASSAGE)
                coupe = coupe if coupe > 0 else TAILLE_PASSAGE
                resultat.append(phrase[:coupe])
                phrase = phrase[coupe:].strip()
            if phrase:
                resultat.append(phrase)
    return resultat

FIN_TROP_COURTE = 250

def decouper(texte):
    liste = phrases(texte)
    morceaux, i = [], 0
    while i < len(liste):
        j, longueur = i, 0
        while j < len(liste) and (j == i or longueur + len(liste[j]) + 1 <= TAILLE_PASSAGE):
            longueur += len(liste[j]) + 1
            j += 1
        # Une fin de texte trop courte est collée à ce passage : seule, BM25 la ferait passer devant
        # les passages complets (il favorise les textes courts)
        if j < len(liste) and sum(len(p) + 1 for p in liste[j:]) < FIN_TROP_COURTE:
            j = len(liste)
        morceaux.append(' '.join(liste[i:j]))
        if j >= len(liste):
            break
        # On reprend les dernières phrases seulement si la phrase suivante tient encore à côté : sinon le
        # passage ne contiendrait que la reprise
        k, recul = j, 0
        while k - 1 > i and recul < CHEVAUCHEMENT and recul + len(liste[k - 1]) + len(liste[j]) + 2 <= TAILLE_PASSAGE:
            k -= 1
            recul += len(liste[k]) + 1
        i = k
    return morceaux

# Le titre du document en tête de chaque passage : au milieu d'un article, le modèle doit savoir de quoi on parle
def avec_titre(titre, morceau):
    if titre.startswith('http') or normaliser(morceau[:len(titre) + 4]).find(normaliser(titre)) >= 0:
        return morceau
    return f"{titre} — {morceau}"

def domaine(url):
    return re.sub(r'^https?://(www\.)?', '', url or '').split('/')[0]

def passages_wikipedia():
    with open(chemins.MARKETING_WIKIPEDIA, 'r', encoding='utf-8') as f:
        for article in map(json.loads, f):
            texte = FIN_D_ARTICLE.split(reparer(article['texte']))[0]
            for morceau in decouper(texte):
                if len(morceau) >= LONGUEUR_MINIMALE_PASSAGE:
                    yield {'titre': article['titre'], 'texte': avec_titre(article['titre'], morceau),
                           'source': 'wikipedia', 'url': article['url'], 'doc': f"wikipedia:{article['id']}"}

def passages_web():
    if not os.path.exists(chemins.MARKETING_WEB):
        print("  pas de pages web (lancer collecte/marketing_web.py)")
        return
    with open(chemins.MARKETING_WEB, 'r', encoding='utf-8') as f:
        for page in map(json.loads, f):
            titre = page['titre'] if not page['titre'].startswith('http') else domaine(page['url'])
            for morceau in decouper(page['texte']):
                if len(morceau) >= LONGUEUR_MINIMALE_PASSAGE:
                    yield {'titre': titre, 'texte': avec_titre(titre, morceau), 'source': 'web', 'url': page['url'],
                           'doc': f"web:{page['id']}"}

def lire_fiche(chemin):
    texte = open(chemin, encoding='utf-8').read()
    _, entete, corps = texte.split('---', 2)
    meta = dict(ligne.split(':', 1) for ligne in entete.strip().splitlines())
    meta = {cle.strip(): valeur.strip() for cle, valeur in meta.items()}
    corps = re.sub(r'<!--.*?-->', '', corps, flags=re.S)
    morceaux = re.split(r'^## +(.+)$', corps, flags=re.M)
    sections = {titre.strip(): contenu.strip() for titre, contenu in zip(morceaux[1::2], morceaux[2::2])}
    return meta, sections

# Une fiche est découpée par rubrique (définition, étapes, exemple...), chaque passage dit de quelle fiche
# et de quelle rubrique il vient. Les mots-clés ne sont pas montrés, ils servent seulement à trouver la fiche
def passages_fiches():
    for chemin in sorted(glob.glob(os.path.join(chemins.FICHES, '*.md'))):
        if os.path.basename(chemin).startswith('_'):
            continue
        meta, sections = lire_fiche(chemin)
        mots_cles = sections.pop('Mots-clés', '') + ' ' + meta.get('mots_cles', '')
        for rubrique, contenu in sections.items():
            contenu = re.sub(r'^\s*(?:\d+\.|-)\s+', '', contenu, flags=re.M)
            for morceau in decouper(contenu):
                yield {'titre': meta['titre'], 'texte': f"{meta['titre']} — {rubrique} : {morceau}", 'source': 'fiche',
                       'url': os.path.relpath(chemin).replace('\\', '/'), 'doc': f"fiche:{os.path.basename(chemin)}",
                       'statut': meta.get('statut', 'brouillon_ia'), 'mots_cles': mots_cles.strip()}

def construire():
    debut = time.time()
    compte = {}

    def tout():
        for source, passages in (('fiche', passages_fiches()), ('wikipedia', passages_wikipedia()),
                                 ('web', passages_web())):
            for passage in passages:
                compte[source] = compte.get(source, 0) + 1
                yield passage
    ecrire_index(tout(), chemins.RECHERCHE_MARKETING)
    taille = sum(os.path.getsize(os.path.join(chemins.RECHERCHE_MARKETING, f))
                 for f in os.listdir(chemins.RECHERCHE_MARKETING)) / 1e6
    print(f"Passages par source : {compte}")
    print(f"Index marketing : {taille:,.0f} Mo sur le disque, construit en {time.time() - debut:.0f}s")

# --- Routage entre l'index général et l'index marketing ---
MODES = ('general', 'marketing', 'auto')
MODE_PAR_DEFAUT = 'auto'
# Reciprocal Rank Fusion : chaque index classe ses passages, on additionne 1 / (K_RRF + rang). Ça évite de
# comparer des scores BM25 calculés sur deux index différents
K_RRF = 60
PROFONDEUR = 20
# Une fiche passe devant ce qui est classé jusqu'à ~3 rangs au-dessus d'elle
BONUS_FICHE = 1 / (K_RRF + 1) - 1 / (K_RRF + 4)

class RechercheRoutee:
    def __init__(self, mode=MODE_PAR_DEFAUT):
        if mode not in MODES:
            raise ValueError(f"mode de recherche inconnu : {mode} (choisir parmi {', '.join(MODES)})")
        if mode != 'general' and not os.path.exists(os.path.join(chemins.RECHERCHE_MARKETING, 'positions.npy')):
            print(f"Pas d'index marketing (lancer python recherche_marketing.py) : recherche dans l'index général seul")
            mode = 'general'
        self.mode = mode
        self.index = {}
        if mode in ('general', 'auto'):
            self.index['general'] = Recherche()
        if mode in ('marketing', 'auto'):
            self.index['marketing'] = Recherche(chemins.RECHERCHE_MARKETING)
        self.nb_passages = sum(i.nb_passages for i in self.index.values())

    # Tous les passages trouvés, du meilleur au moins bon, un seul par document. Un article de Wikipédia
    # peut être dans les deux index : on ne garde que son meilleur passage
    def classer(self, message, seuil=0.0, profondeur=PROFONDEUR):
        resultats = []
        for nom, index in self.index.items():
            vus, rang = set(), 0
            for score, numero in index.chercher(message, n=profondeur, seuil=seuil):
                fiche = index.fiche(numero)
                fiche.setdefault('source', 'wikipedia')
                document = fiche.get('doc', f"{nom}:{numero}")
                if document in vus:
                    continue
                vus.add(document)
                rang += 1
                rrf = 1 / (K_RRF + rang) + (BONUS_FICHE if fiche['source'] == 'fiche' else 0)
                resultats.append((rrf, score, dict(fiche, index=nom)))
        resultats.sort(key=lambda r: (-r[0], -r[1]))
        gardes, vus = [], set()
        for _, _, fiche in resultats:
            cle = ('wikipedia', fiche['titre']) if fiche['source'] == 'wikipedia' else fiche.get('doc', id(fiche))
            if cle not in vus:
                vus.add(cle)
                gardes.append(fiche)
        return gardes

    # Le meilleur passage en dernier, juste avant la question, comme Recherche.documents_pour
    def trouver(self, message, n=2, seuil=SEUIL):
        if not est_question_de_culture(message):
            return []
        return self.classer(message, seuil)[:n][::-1]

    def documents_pour(self, message, n=2, seuil=SEUIL):
        return [(d['titre'], d['texte']) for d in self.trouver(message, n, seuil)]

# Ce que l'interface affiche pour chaque passage lu
def etiquette(document):
    if document['source'] == 'fiche':
        return f"{document['titre']} (fiche{', brouillon' if document.get('statut') != 'relu' else ' relue'})"
    if document['source'] == 'web':
        return f"{document['titre'][:60]} ({domaine(document['url'])})"
    return f"{document['titre']} (Wikipédia)"

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    if len(sys.argv) == 1:
        construire()
    else:
        question = ' '.join(sys.argv[1:])
        for mode in MODES:
            routee = RechercheRoutee(mode)
            debut = time.time()
            trouves = routee.trouver(question)
            print(f"\n== {mode} ({(time.time() - debut) * 1000:.0f} ms)")
            for document in reversed(trouves):
                print(f"  {etiquette(document)}\n      {document['texte'][:250]}")
