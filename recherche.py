import array
import json
import math
import os
import re
import sys
import time
import unicodedata
import zlib
import numpy as np
import chemins

# On indexe seulement le début de chaque article (700 caractères max)
LONGUEUR_PASSAGE = 700
LONGUEUR_MINIMALE_PASSAGE = 150

# Réglages classiques de BM25, et les mots du titre comptent triple
k1, b = 1.2, 0.75
POIDS_TITRE = 3

# En dessous de ce score on préfère ne rien donner au modèle qu'un document hors sujet
SEUIL = 0.4

BONUS_TITRE = 0.2
MALUS_HOMONYME = 0.15
NB_CANDIDATS = 20
# À score proche, un grand article passe devant un petit (« Rome » devant les ambassades d'Italie)
POIDS_IMPORTANCE = 0.06

# Chaque mot est rangé dans un seau par hash, pas besoin de garder le vocabulaire en mémoire
NB_SEAUX = 1 << 24

MOTS_VIDES = set("""
le la les l un une des du de d au aux et ou en a est sont etait etaient ete etre
ce cet cette ces se sa son ses qui que qu quoi quel quelle quels quelles dont
comment pourquoi combien quand il elle ils elles on y ne n pas plus par pour sur
dans avec sans sous entre vers chez c s t tu toi te vous nous je j me m moi mon
ma mes ton ta tes votre vos notre nos leur leurs lui eux peux peut sais sait
savoir connais connait explique expliquer parle parler dis dire donne donner
raconte ca va bien merci bonjour salut oui non ok accord faire fait fais avoir as
ai ont es suis etes sommes cela ceci celui celle ceux tout tous toute toutes tres
aussi alors donc mais si ni car comme ete quelqu chose
""".split())

# Une question sur l'utilisateur (« comment je m'appelle ») ou sur l'avis du modèle se répond avec
# la conversation, pas avec Wikipédia. « explique-moi » par contre compte comme une question
PREMIERE_PERSONNE = re.compile(r"\b(je|j|me|m|moi|mon|ma|mes)\b")
DEMANDE_AU_MODELE = re.compile(r"-moi\b|\bm'(expliquer|explique|en dire|aider|apprendre)\b|\bme (dire|parler|raconter|donner|montrer)\b")
AVIS_DU_MODELE = re.compile(r"\btu (aimes|preferes|penses|crois|veux|vas|es|as|fais|ressens|t'appelles)\b")

def normaliser(texte):
    texte = texte.lower().replace('œ', 'oe').replace('æ', 'ae')
    return ''.join(c for c in unicodedata.normalize('NFD', texte) if unicodedata.category(c) != 'Mn')

def mots(texte):
    resultat = []
    for mot in re.findall(r'[a-z0-9]+', normaliser(texte)):
        if mot in MOTS_VIDES:
            continue
        if len(mot) > 3 and mot[-1] in 'sx':
            mot = mot[:-1]
        resultat.append(mot)
    return resultat

def seau(mot):
    return zlib.crc32(mot.encode()) & (NB_SEAUX - 1)

def cle_titre(texte):
    texte = re.sub(r'\s*\(.*\)', '', texte)
    return zlib.crc32(' '.join(re.findall(r'[a-z0-9]+', normaliser(texte))).encode())

DOCUMENT = "Document : "

def ligne_document(passage):
    return DOCUMENT + passage + "\n"

def est_question_de_culture(message):
    texte = DEMANDE_AU_MODELE.sub(' ', normaliser(message).replace('’', "'"))
    return not PREMIERE_PERSONNE.search(texte) and not AVIS_DU_MODELE.search(texte) and bool(mots(message))

PARENTHESES_VIDES = re.compile(r" ?\((?:\s|[,;:.]|\bou\b|\bsoit\b|\ben(?: [a-zéèêëïîôûç]+)?\b|prononcé(?: en français)?)*\)")
ELISION_PERDUE = re.compile(r"\b([LlDd])([AEIOUYÉÈÊÂÎÔÀ])(?=[a-zàâçéèêëîïôûùüÿœ])")
DATE_PERDUE = re.compile(r" le(?<=\w le)" + ''.join(rf"(?<!\b{p} le)" for p in ('vers', 'depuis', 'de', 'dès', 'avant', 'après', 'entre'))
                         + r"(?:(?= à | et )| (?=[,.]))")

# Répare les trous laissés par les modèles de page perdus : « LItalie ( ) est » -> « L'Italie est »
def reparer(texte):
    texte = PARENTHESES_VIDES.sub('', texte)
    return DATE_PERDUE.sub('', ELISION_PERDUE.sub(r"\1'\2", texte))

def debut_article(texte):
    texte = reparer(texte)
    lignes, longueur = [], 0
    for ligne in texte.split('\n'):
        ligne = ligne.strip()
        if not ligne:
            continue
        if lignes and len(ligne) < 80 and not ligne.endswith(('.', '!', '?', ':', '»', ')')):
            break
        lignes.append(ligne)
        longueur += len(ligne) + 1
        if longueur >= LONGUEUR_PASSAGE:
            break
    passage = ' '.join(lignes)
    if len(passage) > LONGUEUR_PASSAGE:
        fin = passage.rfind('. ', 0, LONGUEUR_PASSAGE)
        passage = passage[:fin + 1] if fin > 200 else passage[:LONGUEUR_PASSAGE].rsplit(' ', 1)[0] + '…'
    return passage

# --- Construction d'un index ---
# passages : des dicts avec au moins 'titre' et 'texte' (le reste est gardé tel quel dans passages.jsonl).
# 'longueur_article', s'il est là, sert à l'importance (un grand article passe devant un petit)
def ecrire_index(passages, dossier):
    os.makedirs(dossier, exist_ok=True)
    debut = time.time()

    seaux, docs, frequences = array.array('i'), array.array('i'), array.array('B')
    longueurs, positions, titres = array.array('H'), array.array('q'), array.array('I')
    articles = array.array('i')
    position = 0

    with open(os.path.join(dossier, 'passages.jsonl'), 'wb') as sortie:
        for passage in passages:
            if 'longueur_article' in passage:
                articles.append(passage.pop('longueur_article'))
            octets = (json.dumps(passage, ensure_ascii=False) + '\n').encode('utf-8')
            numero = len(positions)
            positions.append(position)
            titres.append(cle_titre(passage['titre']))
            sortie.write(octets)
            position += len(octets)

            compte = {}
            for mot in mots(passage['titre']):
                compte[mot] = compte.get(mot, 0) + POIDS_TITRE
            # Les mots-clés d'une fiche servent à la trouver, sans être montrés au modèle
            for mot in mots(passage['texte'] + ' ' + passage.get('mots_cles', '')):
                compte[mot] = compte.get(mot, 0) + 1
            for mot, n in compte.items():
                seaux.append(seau(mot))
                docs.append(numero)
                frequences.append(min(n, 255))
            longueurs.append(min(sum(compte.values()), 65535))

            if numero % 100000 == 0:
                print(f"  {numero:,} passages ({time.time() - debut:.0f}s)", flush=True)

    print(f"{len(positions):,} passages, {len(seaux):,} entrées d'index")

    seaux = np.frombuffer(seaux, dtype=np.int32)
    docs = np.frombuffer(docs, dtype=np.int32)
    frequences = np.frombuffer(frequences, dtype=np.uint8)
    debuts = np.zeros(NB_SEAUX + 1, dtype=np.int64)
    np.cumsum(np.bincount(seaux, minlength=NB_SEAUX), out=debuts[1:])

    fichier = lambda nom: os.path.join(dossier, nom)
    docs_tries = np.lib.format.open_memmap(fichier('docs.npy'), mode='w+', dtype=np.int32, shape=docs.shape)
    frequences_triees = np.lib.format.open_memmap(fichier('frequences.npy'), mode='w+', dtype=np.uint8, shape=docs.shape)
    largeur = NB_SEAUX // 16
    for bas in range(0, NB_SEAUX, largeur):
        indices = np.flatnonzero((seaux >= bas) & (seaux < bas + largeur))
        indices = indices[np.argsort(seaux[indices], kind='stable')]
        depart = debuts[bas]
        docs_tries[depart:depart + len(indices)] = docs[indices]
        frequences_triees[depart:depart + len(indices)] = frequences[indices]
    docs_tries.flush()
    frequences_triees.flush()

    np.save(fichier('debuts.npy'), debuts)
    np.save(fichier('longueurs.npy'), np.frombuffer(longueurs, dtype=np.uint16))
    np.save(fichier('positions.npy'), np.frombuffer(positions, dtype=np.int64))
    np.save(fichier('titres.npy'), np.frombuffer(titres, dtype=np.uint32))
    if len(articles):
        np.save(fichier('articles.npy'), np.frombuffer(articles, dtype=np.int32))
    print(f"Index écrit dans {dossier} ({time.time() - debut:.0f}s)")

# L'index général : le début (700 caractères) de chaque article du Wikipédia de Hugging Face
def construire():
    def passages():
        with open(chemins.CORPUS_COLLECTE, 'r', encoding='utf-8') as entree:
            for ligne in entree:
                try:
                    article = json.loads(ligne)
                except json.JSONDecodeError:
                    continue
                if article.get('source') != 'huggingface':
                    continue
                passage = debut_article(article['texte'])
                if len(passage) >= LONGUEUR_MINIMALE_PASSAGE:
                    yield {'titre': article['titre'], 'texte': passage, 'longueur_article': len(article['texte'])}
    ecrire_index(passages(), chemins.RECHERCHE)

# --- Recherche ---
class Recherche:
    def __init__(self, dossier=chemins.RECHERCHE):
        fichier = lambda nom: os.path.join(dossier, nom)
        self.fichier_passages = fichier('passages.jsonl')
        # L'index (545 Mo) reste sur le disque, une recherche prend ~20 ms
        self.debuts = np.load(fichier('debuts.npy'), mmap_mode='r')
        self.docs = np.load(fichier('docs.npy'), mmap_mode='r')
        self.frequences = np.load(fichier('frequences.npy'), mmap_mode='r')
        self.longueurs = np.load(fichier('longueurs.npy')).astype(np.float32)
        self.positions = np.load(fichier('positions.npy'))
        self.nb_passages = len(self.positions)
        self.longueur_moyenne = float(self.longueurs.mean())
        titres = np.load(fichier('titres.npy'))
        self.ordre_titres = np.argsort(titres, kind='stable')
        self.titres_tries = titres[self.ordre_titres]
        if os.path.exists(fichier('articles.npy')):
            longueurs_articles = np.maximum(np.load(fichier('articles.npy')), 2000)
            self.importance = np.log10(longueurs_articles / 2000).astype(np.float32)
        else:
            self.importance = np.zeros(self.nb_passages, dtype=np.float32)

    def par_titre(self, question):
        tous = re.findall(r'[a-z0-9]+', normaliser(question))
        trouves = []
        for taille in range(1, 6):
            for i in range(len(tous) - taille + 1):
                morceau = tous[i:i + taille]
                if all(m in MOTS_VIDES for m in morceau):
                    continue
                cle = zlib.crc32(' '.join(morceau).encode())
                debut = np.searchsorted(self.titres_tries, cle)
                fin = np.searchsorted(self.titres_tries, cle, side='right')
                trouves.extend(int(n) for n in self.ordre_titres[debut:min(fin, debut + 10)])
        return trouves

    def fiche(self, numero):
        with open(self.fichier_passages, 'rb') as f:
            f.seek(int(self.positions[numero]))
            return json.loads(f.readline())

    def passage(self, numero):
        article = self.fiche(numero)
        return article['titre'], article['texte']

    def chercher(self, question, n=1, seuil=0.0):
        scores = np.zeros(self.nb_passages, dtype=np.float32)
        maximum = 0.0
        for mot in set(mots(question)):
            s = seau(mot)
            debut, fin = int(self.debuts[s]), int(self.debuts[s + 1])
            if debut == fin:
                continue
            docs = np.asarray(self.docs[debut:fin])
            frequences = np.asarray(self.frequences[debut:fin], dtype=np.float32)
            importance = math.log(1 + (self.nb_passages - (fin - debut) + 0.5) / (fin - debut + 0.5))
            norme = k1 * (1 - b + b * self.longueurs[docs] / self.longueur_moyenne)
            np.add.at(scores, docs, importance * frequences * (k1 + 1) / (frequences + norme))
            maximum += importance * (k1 + 1)
        if maximum == 0:
            return []
        scores /= maximum

        classement = np.where(scores > 0, scores + POIDS_IMPORTANCE * self.importance, 0)

        nb = min(max(n, NB_CANDIDATS), self.nb_passages - 1)
        candidats = set(np.argpartition(-classement, nb)[:nb].tolist()) | set(self.par_titre(question))
        mots_question = set(mots(question))
        resultats = []
        for i in candidats:
            if scores[i] <= 0:
                continue
            titre = self.passage(i)[0]
            score = float(scores[i])
            if set(mots(re.sub(r'\s*\(.*\)', '', titre))) <= mots_question:
                score += BONUS_TITRE
            if '(' in titre:
                score -= MALUS_HOMONYME
            if score >= seuil:
                resultats.append((score + POIDS_IMPORTANCE * float(self.importance[i]), score, int(i)))
        return [(score, i) for _, score, i in sorted(resultats, reverse=True)[:n]]

    def documents_pour(self, message, n=2, seuil=SEUIL):
        if not est_question_de_culture(message):
            return []
        # Le meilleur passage en dernier, juste avant la question : c'est celui que le modèle recopie le plus
        return [self.passage(i) for _, i in self.chercher(message, n, seuil=seuil)][::-1]

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    if len(sys.argv) == 1:
        construire()
    else:
        recherche = Recherche()
        question = ' '.join(sys.argv[1:])
        print(f"question de culture : {est_question_de_culture(question)}, seuil {SEUIL}")
        for score, numero in recherche.chercher(question, n=3):
            titre, texte = recherche.passage(numero)
            print(f"\n{score:.2f}  {titre}\n      {texte[:300]}")
