import argparse
import array
import collections
import gzip
import json
import os
import re
import sys
import time

import numpy as np
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chemins

# python collecte/marketing_wikipedia.py            -> arbre des catégories et rapport, sans rien écrire
# python collecte/marketing_wikipedia.py --ecrire   -> écrit donnees/marketing/wikipedia.jsonl

# Le Wikipédia de Hugging Face n'a pas les catégories : on les prend dans les tables SQL de Wikimedia.
# Depuis 2025 categorylinks ne donne plus le nom de la catégorie mais un numéro dans linktarget
DUMPS = 'https://dumps.wikimedia.org/frwiki/latest/frwiki-latest-{}.sql.gz'
TABLES = ('page', 'linktarget', 'categorylinks')
# Wikimedia refuse le User-Agent par défaut de requests (403)
ENTETES = {'User-Agent': 'EDP-LLM/1.0 (projet etudiant, modele de langage)'}

# « Commerce électronique » n'existe pas sur le Wikipédia français, c'est « Commerce en ligne ».
# Les six dernières racines ont été ajoutées après relecture de l'arbre
RACINES = ["Marketing", "Publicité", "Communication", "Relations publiques", "Vente", "Commerce en ligne",
           "Marque", "Gestion de la relation client", "Référencement", "Stratégie d'entreprise",
           "Étude de marché", "Comportement du consommateur",
           "Terminologie du marketing", "Fidélisation", "Merchandising", "Comparateur de prix", "Site de réseau social",
           "Sondage"]
PROFONDEUR_MAX = 3
# Branches coupées (avec tout ce qu'il y a dessous) après relecture de l'arbre : elles dérivent vers la
# littérature, les médias, le droit, des fiches d'entreprises ou de marques
EXCLUSIONS = [
    # Communication : on garde ses articles directs et ce qui touche à l'entreprise
    "Littérature", "Médias", "Science de l'information", "Langue", "Œuvre d'art",
    "Sciences de l'information et de la communication", "Signal visuel", "Langage", "Communication humaine",
    "Information", "Graphisme", "Communication politique", "Communication scientifique", "Mensonge",
    "Film sur la communication", "Signalisation", "Communication par pays", "Biais", "Fuite d'information",
    "Transmissions militaires", "Signal sonore", "Transcommunication", "Bouteille à la mer", "Congrès annuel",
    "Chercheur en communication", "Chercheuse en communication", "Communication par continent",
    "Ministère de la Communication", "Signal",
    # Marketing
    "Route commerciale", "Zone d'aménagement concerté", "Distribution de jeux vidéo", "Film commandité",
    "Conditionnement",
    # Publicité
    "Bière et sponsoring", "Média sans publicité",
    # Vente
    "Lobbying", "Contrebande", "École de commerce", "Vente aux enchères",
    # Stratégie d'entreprise
    "Droit de la concurrence", "Pure player", "Acquisition d'entreprise", "Gouvernance d'entreprise",
    "Sous-traitance",
    # Comportement du consommateur
    "Cuisine de rue",
    # Marque : des milliers de fiches de marques, pas du marketing
    "Marque par pays", "Marque par secteur économique", "Marque par entreprise", "Marque générique",
    "Spécialité pharmaceutique", "Vin de marque", "Marque de fiction",
    # Sondage
    "Sondages préélectoraux",
    # Deuxième relecture : modes et fiches de marques qui reviennent par d'autres chemins
    "Phénomène de mode", "Marque française", "Site web sans publicité",
    "Pavillon bleu", "Film sur la publicité", "Holding", "Chaîne de télévision de télé-achat",
]
# Catégories de maintenance de Wikipédia (ébauches, portails, projets), jamais du contenu
MAINTENANCE = ('Wikipédia:', 'Portail:', 'Projet:', 'Article_', 'Utilisateur', 'Page_')

LONGUEUR_MINIMALE = 1500
ESPACE_CATEGORIE = 14

# --- Téléchargement, avec reprise si la connexion coupe ---
def telecharger(table):
    os.makedirs(chemins.DUMPS_WIKIPEDIA, exist_ok=True)
    fichier = os.path.join(chemins.DUMPS_WIKIPEDIA, f'frwiki-latest-{table}.sql.gz')
    url = DUMPS.format(table)
    total = int(requests.head(url, headers=ENTETES, timeout=60).headers['Content-Length'])
    partiel = fichier + '.partiel'
    if os.path.exists(fichier) and os.path.getsize(fichier) == total:
        return fichier
    deja = os.path.getsize(partiel) if os.path.exists(partiel) else 0
    debut = time.time()
    with requests.get(url, headers={**ENTETES, 'Range': f'bytes={deja}-'}, stream=True, timeout=60) as reponse, \
         open(partiel, 'ab') as sortie:
        reponse.raise_for_status()
        for bloc in reponse.iter_content(1 << 20):
            sortie.write(bloc)
            deja += len(bloc)
            if deja % (100 << 20) < (1 << 20):
                print(f"  {table} : {deja / 1e6:,.0f} / {total / 1e6:,.0f} Mo ({time.time() - debut:.0f}s)", flush=True)
    os.replace(partiel, fichier)
    return fichier

# --- Lecture des tables SQL en flux, une ligne = un tuple ---
ECHAPPEMENTS = {b'0': b'\0', b'n': b'\n', b'r': b'\r', b'Z': b'\x1a'}

def deschapper(texte):
    return re.sub(rb'\\(.)', lambda m: ECHAPPEMENTS.get(m.group(1), m.group(1)), texte).decode('utf-8', 'replace')

def lignes(table):
    with gzip.open(os.path.join(chemins.DUMPS_WIKIPEDIA, f'frwiki-latest-{table}.sql.gz'), 'rb') as f:
        for ligne in f:
            if ligne.startswith(b'('):
                yield ligne.rstrip()

PAGE = re.compile(rb"^\((\d+),(-?\d+),'((?:[^'\\]|\\.)*)',")
CIBLE = re.compile(rb"^\((\d+),(-?\d+),'((?:[^'\\]|\\.)*)'\)[,;]?$")
LIEN = re.compile(rb"^\((\d+),.*,'(page|subcat|file)',\d+,(\d+)\)[,;]?$")
TYPES = {b'page': 0, b'subcat': 1, b'file': 2}

def lire_tables():
    cache = os.path.join(chemins.DUMPS_WIKIPEDIA, 'tables.npz')
    if os.path.exists(cache):
        tables = np.load(cache, allow_pickle=False)
        noms = json.load(open(cache.replace('.npz', '.json'), encoding='utf-8'))
        return (dict(zip(tables['pages'].tolist(), noms['pages'])), dict(zip(tables['cibles'].tolist(), noms['cibles'])),
                tables['de'], tables['vers'], tables['type'])
    debut = time.time()
    # Le nom de chaque page de catégorie : une sous-catégorie est rangée par son numéro de page
    pages = {}
    for ligne in lignes('page'):
        m = PAGE.match(ligne)
        if m and int(m.group(2)) == ESPACE_CATEGORIE:
            pages[int(m.group(1))] = deschapper(m.group(3))
    print(f"  page : {len(pages):,} catégories ({time.time() - debut:.0f}s)", flush=True)
    cibles = {}
    for ligne in lignes('linktarget'):
        m = CIBLE.match(ligne)
        if m and int(m.group(2)) == ESPACE_CATEGORIE:
            cibles[int(m.group(1))] = deschapper(m.group(3))
    print(f"  linktarget : {len(cibles):,} catégories ({time.time() - debut:.0f}s)", flush=True)
    # ~25 millions de liens : des array plutôt que des listes d'int Python (10 fois moins de mémoire)
    de, vers, types = array.array('i'), array.array('q'), array.array('B')
    for n, ligne in enumerate(lignes('categorylinks')):
        m = LIEN.match(ligne)
        if m:
            de.append(int(m.group(1)))
            types.append(TYPES[m.group(2)])
            vers.append(int(m.group(3)))
        if n % 5_000_000 == 0:
            print(f"  categorylinks : {n:,} liens ({time.time() - debut:.0f}s)", flush=True)
    de, vers, types = np.frombuffer(de, dtype=np.int32), np.frombuffer(vers, dtype=np.int64), np.frombuffer(types, dtype=np.uint8)
    np.savez(cache, pages=np.array(list(pages), dtype=np.int64), cibles=np.array(list(cibles), dtype=np.int64),
             de=de, vers=vers, type=types)
    with open(cache.replace('.npz', '.json'), 'w', encoding='utf-8') as f:
        json.dump({'pages': list(pages.values()), 'cibles': list(cibles.values())}, f, ensure_ascii=False)
    print(f"  {len(de):,} liens de catégorie lus ({time.time() - debut:.0f}s)", flush=True)
    return pages, cibles, de, vers, types

# --- Arbre des catégories ---
def parcourir(enfants, existantes):
    retenues = {}
    a_voir = collections.deque()
    for racine in RACINES:
        titre = racine.replace(' ', '_')
        if titre in existantes:
            retenues[titre] = (0, None, titre)
            a_voir.append(titre)
        else:
            print(f"  catégorie racine introuvable : {racine}")
    exclues = {e.replace(' ', '_') for e in EXCLUSIONS}
    # En largeur : une catégorie atteinte par deux chemins garde le plus court, et on ne repasse
    # jamais par une catégorie déjà vue, ce qui coupe les cycles
    while a_voir:
        titre = a_voir.popleft()
        profondeur, _, racine = retenues[titre]
        if profondeur == PROFONDEUR_MAX:
            continue
        for enfant in sorted(enfants.get(titre, ())):
            if enfant in retenues or enfant in exclues or enfant.startswith(MAINTENANCE):
                continue
            retenues[enfant] = (profondeur + 1, titre, racine)
            a_voir.append(enfant)
    return retenues

# --- Articles ---
PREFIXE = re.compile(r'^\{"source": "huggingface", "id": "(\d+)"')
PERSONNE = re.compile(r"\bnée?\b")
OEUVRE = re.compile(r"\((?:film|série|téléfilm|roman|album|chanson|jeu vidéo|émission|bande dessinée|livre)\b")
CATEGORIES_PERSONNE = ('Naissance_', 'Décès_')

def decrire(article, garder_texte):
    return {'titre': article['titre'], 'longueur': len(article['texte']),
            'personne': bool(PERSONNE.search(article['texte'][:250])), 'url': article.get('url', ''),
            'texte': article['texte'] if garder_texte else None}

def lire_articles(ids, garder_texte):
    articles = {}
    with open(chemins.CORPUS_COLLECTE, 'r', encoding='utf-8') as f:
        for ligne in f:
            m = PREFIXE.match(ligne)
            if not m or int(m.group(1)) not in ids:
                continue
            articles[int(m.group(1))] = decrire(json.loads(ligne), garder_texte)
    # Les articles de moins de 2 000 caractères, récupérés à part avec --completer
    if os.path.exists(chemins.WIKIPEDIA_COMPLEMENT):
        with open(chemins.WIKIPEDIA_COMPLEMENT, 'r', encoding='utf-8') as f:
            for article in map(json.loads, f):
                if article['id'] in ids and article['id'] not in articles:
                    articles[article['id']] = decrire(article, garder_texte)
    return articles

# corpus_brut.jsonl a laissé de côté les articles de moins de 2 000 caractères : on va les chercher dans le même
# Wikipédia de Hugging Face. On lit d'abord la seule colonne id de chaque groupe de lignes, et le texte
# seulement là où il y a un article qui manque
def completer(manquants):
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, HfFileSystem
    from collecte_commun import nettoyer
    jeu, config = 'wikimedia/wikipedia', '20231101.fr'
    a_trouver = {str(i) for i in manquants}
    if os.path.exists(chemins.WIKIPEDIA_COMPLEMENT):
        with open(chemins.WIKIPEDIA_COMPLEMENT, 'r', encoding='utf-8') as f:
            a_trouver -= {str(json.loads(l)['id']) for l in f}
    debut, trouves = time.time(), 0
    fichiers = sorted(f.rfilename for f in HfApi().dataset_info(jeu).siblings if f.rfilename.startswith(config + '/'))
    with open(chemins.WIKIPEDIA_COMPLEMENT, 'a', encoding='utf-8') as sortie:
        for fichier in fichiers:
            if not a_trouver:
                break
            with HfFileSystem().open(f'datasets/{jeu}/{fichier}', block_size=8 << 20) as h:
                parquet = pq.ParquetFile(h)
                for groupe in range(parquet.num_row_groups):
                    if not a_trouver.intersection(parquet.read_row_group(groupe, columns=['id']).column('id').to_pylist()):
                        continue
                    for ligne in parquet.read_row_group(groupe, columns=['id', 'title', 'text', 'url']).to_pylist():
                        if ligne['id'] in a_trouver:
                            sortie.write(json.dumps({'id': int(ligne['id']), 'titre': ligne['title'],
                                                     'texte': nettoyer(ligne['text']), 'url': ligne['url']},
                                                    ensure_ascii=False) + '\n')
                            a_trouver.discard(ligne['id'])
                            trouves += 1
            print(f"  {fichier.rsplit('/', 1)[-1]} : {trouves:,} articles récupérés ({time.time() - debut:.0f}s)", flush=True)
    print(f"{trouves:,} articles récupérés, {len(a_trouver):,} absents du dump de novembre 2023 (créés depuis)")

def taille(chemin):
    return sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(chemin) for f in fs) / 1e6

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ecrire', action='store_true', help="écrit wikipedia.jsonl (sinon rapport seul)")
    parser.add_argument('--completer', action='store_true',
                        help="va chercher les articles de moins de 2 000 caractères absents de corpus_brut.jsonl")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    debut = time.time()

    for table in TABLES:
        telecharger(table)
    print(f"Tables téléchargées : {taille(chemins.DUMPS_WIKIPEDIA):,.0f} Mo ({time.time() - debut:.0f}s)")

    pages, cibles, de, vers, types = lire_tables()
    titre_vers_cible = {titre: i for i, titre in cibles.items()}
    enfants = collections.defaultdict(set)
    for parent, enfant in zip(vers[types == 1].tolist(), de[types == 1].tolist()):
        if parent in cibles and enfant in pages:
            enfants[cibles[parent]].add(pages[enfant])
    retenues = parcourir(enfants, set(titre_vers_cible) | set(enfants))
    print(f"{len(retenues):,} catégories retenues ({time.time() - debut:.0f}s)")

    # Les articles de chaque catégorie, rangés dans la moins profonde où on les trouve
    numeros = {titre_vers_cible[t]: t for t in retenues if t in titre_vers_cible}
    membres = (types == 0) & np.isin(vers, list(numeros))
    rangement = {}
    for article, categorie in zip(de[membres].tolist(), vers[membres].tolist()):
        titre = numeros[categorie]
        if article not in rangement or retenues[titre][0] < retenues[rangement[article]][0]:
            rangement[article] = titre
    candidats = np.array(list(rangement), dtype=np.int32)

    # Biographies et homonymies d'après leurs catégories, listes d'après le titre
    def dans(prefixes):
        marquees = [i for i, t in cibles.items() if t.startswith(prefixes)]
        lignes_ = (types == 0) & np.isin(de, candidats) & np.isin(vers, marquees)
        return set(de[lignes_].tolist())
    personnes, homonymies = dans(CATEGORIES_PERSONNE), dans(('Homonymie',))
    articles = lire_articles(set(rangement), args.ecrire)
    if args.completer:
        completer([n for n in rangement if n not in articles])
        articles = lire_articles(set(rangement), args.ecrire)
    print(f"{len(rangement):,} articles dans ces catégories, {len(articles):,} présents dans le corpus "
          f"({time.time() - debut:.0f}s)")

    raisons = collections.Counter()
    gardes = {}
    for numero, titre_categorie in rangement.items():
        article = articles.get(numero)
        if article is None:
            raisons['absent (créé après novembre 2023, ou moins de 2 000 caractères sans --completer)'] += 1
        elif numero in personnes or article['personne']:
            raisons['biographie'] += 1
        elif numero in homonymies or '(homonymie)' in article['titre']:
            raisons['homonymie'] += 1
        elif re.match(r"Listes? ", article['titre']):
            raisons['liste'] += 1
        elif OEUVRE.search(article['titre']):
            raisons['œuvre (film, série, livre...)'] += 1
        elif article['longueur'] < LONGUEUR_MINIMALE:
            raisons[f'moins de {LONGUEUR_MINIMALE} caractères'] += 1
        else:
            gardes[numero] = titre_categorie
    print(f"\n{len(gardes):,} articles gardés. Rejetés : {dict(raisons)}")

    # --- Rapport : l'arbre, avec le nombre d'articles gardés dans chaque catégorie ---
    par_categorie = collections.Counter(gardes.values())
    sous_arbre = collections.Counter()
    for titre in retenues:
        t = titre
        while t is not None:
            sous_arbre[t] += par_categorie[titre]
            t = retenues[t][1]
    rapport = {t: {'profondeur': p, 'parent': parent, 'racine': racine, 'articles': par_categorie[t],
                   'articles_dessous': sous_arbre[t]} for t, (p, parent, racine) in retenues.items()}
    with open(chemins.CATEGORIES_MARKETING, 'w', encoding='utf-8') as f:
        json.dump(rapport, f, ensure_ascii=False, indent=1)
    with open(chemins.CATEGORIES_MARKETING.replace('.json', '.txt'), 'w', encoding='utf-8') as f:
        def ecrire(titre):
            r = rapport[titre]
            f.write(f"{'    ' * r['profondeur']}{titre.replace('_', ' ')}  [{r['articles']} / {r['articles_dessous']}]\n")
            for enfant in sorted((e for e in enfants.get(titre, ()) if e in rapport and rapport[e]['parent'] == titre),
                                 key=lambda e: -rapport[e]['articles_dessous']):
                ecrire(enfant)
        for racine in RACINES:
            if racine.replace(' ', '_') in rapport:
                ecrire(racine.replace(' ', '_'))
    for racine in RACINES:
        t = racine.replace(' ', '_')
        if t in rapport:
            nb = sum(1 for r in rapport.values() if r['racine'] == t)
            print(f"  {racine:32} {nb:5} catégories, {rapport[t]['articles_dessous']:6,} articles")
    print(f"Arbre détaillé : {chemins.CATEGORIES_MARKETING.replace('.json', '.txt')}  "
          f"[articles dans la catégorie / dans tout son sous-arbre]")

    if args.ecrire:
        with open(chemins.MARKETING_WIKIPEDIA, 'w', encoding='utf-8') as f:
            for numero, titre_categorie in gardes.items():
                a = articles[numero]
                f.write(json.dumps({'source': 'wikipedia', 'id': numero, 'titre': a['titre'], 'url': a['url'],
                                    'categorie': titre_categorie.replace('_', ' '), 'texte': a['texte']},
                                   ensure_ascii=False) + '\n')
        print(f"{len(gardes):,} articles -> {chemins.MARKETING_WIKIPEDIA} "
              f"({os.path.getsize(chemins.MARKETING_WIKIPEDIA) / 1e6:.0f} Mo)")
    print(f"Terminé en {time.time() - debut:.0f}s")

if __name__ == '__main__':
    main()
