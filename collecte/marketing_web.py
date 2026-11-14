import argparse
import collections
import hashlib
import json
import os
import random
import re
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chemins
from recherche import normaliser

# python collecte/marketing_web.py --echantillon 200000   -> note un échantillon pour régler le seuil
# python collecte/marketing_web.py                        -> passe complète (reprend où elle s'était arrêtée)

# FineWeb-2, partie française (licence ODC-By 1.0). 135 fichiers parquet de ~4,85 Go, lus en flux
JEU = 'HuggingFaceFW/fineweb-2'
CONFIG = 'fra_Latn'
DOSSIER_HF = 'data/fra_Latn/train'

PLAFOND = 200_000
LONGUEUR_MIN, LONGUEUR_MAX = 1500, 30000

# --- Passe 1 : score de mots-clés marketing ---
# Poids 3 : sans ambiguïté. 2 : très marketing mais pas que. 1 : courant ailleurs (« campagne », « cible »)
MOTS_CLES = {
    3: ["marketing", "webmarketing", "persona", "personas", "tunnel de conversion", "entonnoir de conversion",
        "taux de conversion", "referencement naturel", "referencement payant", "emailing", "e-mailing",
        "fidelisation", "cout d'acquisition", "lifetime value", "valeur vie client", "a/b test", "a/b testing",
        "test a/b", "tests a/b", "branding", "community manager", "community management", "copywriting",
        "landing page", "landing pages", "page d'atterrissage", "inbound", "growth hacking", "generation de leads",
        "google ads", "adwords", "facebook ads", "meta ads", "taux d'ouverture", "taux de clic", "parcours client",
        "programme de fidelite", "retargeting", "remarketing", "call to action", "proposition de valeur",
        "brand content", "marketing automation", "lead nurturing", "etude de marche", "plan de communication",
        "strategie de contenu", "calendrier editorial", "marketing d'influence", "storytelling", "mix marketing",
        "plan marketing", "positionnement marketing", "notoriete de marque", "image de marque", "growth marketing"],
    2: ["newsletter", "reseaux sociaux", "positionnement", "ciblage", "leads", "lead", "prospection", "prospects",
        "crm", "experience client", "relation client", "influenceur", "influenceurs", "benchmark", "swot",
        "veille concurrentielle", "avantage concurrentiel", "segmentation", "persona", "tunnel", "conversion",
        "acquisition", "fidelite", "publicitaire", "publicite", "annonceur", "annonceurs", "audience",
        "engagement", "contenu", "abonnes", "e-commerce", "ecommerce", "boutique en ligne", "site web"],
    1: ["campagne", "campagnes", "cible", "cibles", "marque", "marques", "clients", "client", "consommateur",
        "consommateurs", "vente", "ventes", "communication", "strategie", "trafic", "visibilite", "notoriete",
        "budget", "concurrents", "entreprise", "produit", "produits", "offre"],
}
# Sigles : on les cherche avec leur casse (« ROI » et pas « roi », « CAC » mais pas « CAC 40 »)
SIGLES = {3: [r"SEO", r"SEA", r"KPIs?", r"CTR", r"CPC", r"CPM", r"LTV", r"CAC(?! ?40)"],
          2: [r"ROI", r"CRM", r"B2B", r"B2C", r"CTA", r"SMS"]}

def regex_mots(mots):
    return re.compile(r"(?<![a-z0-9])(" + '|'.join(sorted(map(re.escape, mots), key=len, reverse=True)) + r")(?![a-z0-9])")

MOTIFS = [(poids, regex_mots(mots)) for poids, mots in MOTS_CLES.items()]
MOTIFS_SIGLES = [(poids, re.compile(r"\b(" + '|'.join(s) + r")\b")) for poids, s in SIGLES.items()]
# Un premier tri très rapide : une page sans aucun de ces mots n'est même pas notée
INDICE = re.compile(r"marketing|seo\b|referencement|emailing|newsletter|fidelisation|conversion|branding|persona|"
                    r"community|landing|inbound|crm\b|kpi|campagne|publicit|prospect|lead|influen", re.I)

def score(texte, normal):
    poids, distincts = 0, set()
    for p, motif in MOTIFS:
        for m in motif.findall(normal):
            poids += p
            distincts.add(m)
    for p, motif in MOTIFS_SIGLES:
        for m in motif.findall(texte):
            poids += p
            distincts.add(m)
    nb_mots = max(len(normal.split()), 1)
    # Densité pour 100 mots. Un seul mot répété (« newsletter » en pied de page) ne suffit pas
    return poids * 100 / nb_mots, len(distincts)

# --- Passe 2 : qualité ---
INTERDITS = regex_mots(["casino", "casinos", "poker", "paris sportifs", "pari sportif", "machines a sous", "bitcoin",
                        "crypto", "cryptomonnaie", "cryptomonnaies", "ethereum", "nft", "forex", "trading", "porno",
                        "pornographique", "sexy", "escort", "viagra", "cialis", "rencontre coquine",
                        "rencontres coquines", "xxx", "voyance", "marabout", "pret entre particuliers"])
PRIX = re.compile(r"\d+(?:[,.]\d{2})? ?€")
BOUTIQUE = regex_mots(["ajouter au panier", "en stock", "livraison gratuite", "livraison offerte", "rupture de stock",
                       "prix barre", "promo", "code promo", "quantite"])
MOTS_FRANCAIS = set("le la les de des du un une et est en que qui pour dans sur pas par plus au aux il elle nous vous "
                    "ce cette son sa ses leur leurs avec sont ont mais ou comme".split())

# Mentions légales, confidentialité et offres d'emploi parlent d'entreprise sans parler de marketing
PAGE_ADMINISTRATIVE = re.compile(r"mentions?[-_ ]legales|confidentialit|privacy|cookies?|/cgv|/cgu|conditions[-_]generales|"
                                 r"indeed\.|/emploi|/jobs?/|/viewjob|offre[-_]d[-_]emploi|/recrutement|/careers?/")
EMPLOI = regex_mots(["offre d'emploi", "type de contrat", "cdi", "cdd", "poste a pourvoir", "envoyez votre cv",
                     "profil recherche", "nous recrutons", "h/f", "f/h", "cabinet de recrutement"])
# Communiqués et dépêches financières : beaucoup de vocabulaire d'entreprise, rien de réutilisable
COMMUNIQUE = re.compile(r"prnewswire|businesswire|globenewswire|boursier\.com|zonebourse|/communiques?[-_/]|"
                        r"communique[-_]de[-_]presse")
DEPECHE = regex_mots(["communique de presse", "a annonce aujourd'hui", "annonce aujourd'hui", "chiffre d'affaires trimestriel",
                      "resultats annuels", "resultats semestriels", "otc", "nasdaq", "nyse", "euronext"])

def qualite(texte, normal, exemple):
    if not LONGUEUR_MIN <= len(texte) <= LONGUEUR_MAX:
        return 'longueur'
    url = normaliser(exemple.get('url', ''))
    if PAGE_ADMINISTRATIVE.search(url) or len(EMPLOI.findall(normal)) >= 2:
        return 'page légale ou offre d\'emploi'
    if COMMUNIQUE.search(url) or len(DEPECHE.findall(normal)) >= 2:
        return 'communiqué ou dépêche financière'
    lignes = [l.strip() for l in texte.split('\n') if l.strip()]
    if sum(len(l) < 30 for l in lignes) > 0.5 * len(lignes):
        return 'lignes courtes (menus)'
    longues = [l for l in lignes if len(l) >= 60]
    if not longues or sum(l[-1] in '.!?:»)' for l in longues) < 0.5 * len(longues):
        return 'ponctuation'
    mots = normal.split()
    if exemple.get('language_score', 1) < 0.9 or sum(m in MOTS_FRANCAIS for m in mots) < 0.2 * len(mots):
        return 'pas assez de français'
    if len(INTERDITS.findall(normal)) >= 2:
        return 'casino, crypto ou adulte'
    if len(PRIX.findall(texte)) > 8 or len(BOUTIQUE.findall(normal)) >= 3:
        return 'liste de produits'
    pleins = [m for m in mots if m not in MOTS_FRANCAIS and len(m) > 3]
    if pleins and collections.Counter(pleins).most_common(1)[0][1] > 0.06 * len(pleins):
        return 'spam (mot répété)'
    if len(set(lignes)) < 0.8 * len(lignes):
        return 'spam (lignes répétées)'
    return None

# --- Doublons ---
def texte_normalise(normal):
    return ' '.join(re.findall(r'[a-z0-9]+', normal))

NB_HACHAGES, NB_BANDES = 128, 16
PREMIER = (1 << 61) - 1
_alea = np.random.default_rng(12345)
A = _alea.integers(1, PREMIER, NB_HACHAGES, dtype=np.uint64)
B = _alea.integers(0, PREMIER, NB_HACHAGES, dtype=np.uint64)

# Signature MinHash sur les morceaux de 5 mots : deux pages dont 80 % des morceaux sont communs
# ont ~80 % de valeurs égales dans leur signature
def signature(mots_):
    morceaux = {' '.join(mots_[i:i + 5]) for i in range(max(len(mots_) - 4, 1))}
    h = np.array([zlib.crc32(m.encode()) for m in morceaux], dtype=np.uint64)
    # (a * h + b) mod p, en restant sous 2^64 : h tient sur 32 bits, a est réduit à 31 bits
    valeurs = ((A[None, :] >> np.uint64(30)) * h[:, None] + B[None, :]) % np.uint64(PREMIER)
    return valeurs.min(axis=0).astype(np.uint64)

class Doublons:
    def __init__(self):
        self.exacts = set()
        self.bandes = [dict() for _ in range(NB_BANDES)]
        self.signatures = []

    def bandes_de(self, sig):
        largeur = NB_HACHAGES // NB_BANDES
        return [hash(sig[i * largeur:(i + 1) * largeur].tobytes()) for i in range(NB_BANDES)]

    def est_doublon(self, empreinte, sig):
        if empreinte in self.exacts:
            return 'doublon exact'
        candidats = {self.bandes[i].get(b) for i, b in enumerate(self.bandes_de(sig))} - {None}
        for c in candidats:
            if np.mean(self.signatures[c] == sig) >= 0.8:
                return 'quasi-doublon'
        return None

    def ajouter(self, empreinte, sig):
        self.exacts.add(empreinte)
        numero = len(self.signatures)
        self.signatures.append(sig)
        for i, b in enumerate(self.bandes_de(sig)):
            self.bandes[i].setdefault(b, numero)

# --- Lecture de FineWeb-2 ---
def fichiers_du_jeu():
    from huggingface_hub import HfApi
    noms = sorted(f.rfilename for f in HfApi().dataset_info(JEU, files_metadata=False).siblings
                  if f.rfilename.startswith(DOSSIER_HF) and f.rfilename.endswith('.parquet'))
    # Les fichiers suivent l'ordre des crawls : on les prend en sautant, pour mélanger les années
    pas = 7
    return [noms[i] for depart in range(pas) for i in range(depart, len(noms), pas)]

COLONNES = ['text', 'id', 'url', 'date', 'language_score']

# pyarrow directement, par groupes de ~1000 lignes et seulement les colonnes utiles : 2 à 3 fois plus rapide
# que load_dataset en flux
def lire(fichier):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    with HfFileSystem().open(f'datasets/{JEU}/{fichier}', block_size=8 << 20) as h:
        parquet = pq.ParquetFile(h)
        for groupe in range(parquet.num_row_groups):
            yield from parquet.read_row_group(groupe, columns=COLONNES).to_pylist()

def titre_de(texte, url):
    premiere = texte.strip().split('\n', 1)[0].strip()
    return premiere if 10 <= len(premiere) <= 120 else url

def examiner(exemple):
    texte = exemple['text']
    if not INDICE.search(texte):
        return None, 0.0, 0, None
    normal = normaliser(texte)
    densite, distincts = score(texte, normal)
    return normal, densite, distincts, qualite(texte, normal, exemple)

# --- Échantillon pour régler le seuil ---
def echantillon(nombre):
    sortie = os.path.join(chemins.MARKETING, 'web_echantillon.jsonl')
    fichiers = fichiers_du_jeu()
    debut, vus, notes = time.time(), 0, 0
    with open(sortie, 'w', encoding='utf-8') as f:
        # Un peu de chaque époque : 3 fichiers éloignés, nombre / 3 pages dans chacun
        for fichier in (fichiers[0], fichiers[len(fichiers) // 2], fichiers[-30]):
            for n, exemple in enumerate(lire(fichier)):
                if n >= nombre // 3:
                    break
                vus += 1
                normal, densite, distincts, defaut = examiner(exemple)
                if normal is not None and densite > 0:
                    notes += 1
                    f.write(json.dumps({'url': exemple['url'], 'date': exemple['date'], 'densite': round(densite, 3),
                                        'distincts': distincts, 'defaut': defaut, 'texte': exemple['text']},
                                       ensure_ascii=False) + '\n')
                if vus % 20000 == 0:
                    print(f"  {vus:,} pages lues, {notes:,} notées ({vus / (time.time() - debut):.0f} pages/s)", flush=True)
    print(f"{vus:,} pages lues en {time.time() - debut:.0f}s -> {sortie}")

def montrer(seuil, distincts_min):
    pages = [json.loads(l) for l in open(os.path.join(chemins.MARKETING, 'web_echantillon.jsonl'), encoding='utf-8')]
    bonnes = [p for p in pages if p['defaut'] is None]
    for s in (0.5, 1, 1.5, 2, 2.5, 3, 4, 5):
        n = sum(p['densite'] >= s and p['distincts'] >= distincts_min for p in bonnes)
        print(f"  seuil {s:4} : {n:6,} pages gardées sur {len(pages):,} notées")
    print("  défauts de qualité :", collections.Counter(p['defaut'] for p in pages).most_common())
    gardees = [p for p in bonnes if p['densite'] >= seuil and p['distincts'] >= distincts_min]
    justesse = sorted((p for p in bonnes if seuil * 0.75 <= p['densite'] < seuil and p['distincts'] >= distincts_min),
                      key=lambda p: -p['densite'])
    alea = random.Random(0)
    return alea.sample(gardees, min(20, len(gardees))), alea.sample(justesse, min(10, len(justesse)))

# --- Passe complète : plusieurs fichiers en parallèle, reprise après une coupure ---
# Réglé sur un échantillon de 150 000 pages : à 5, environ 3 pages gardées sur 4 sont utiles,
# et 2 pages sur 10 juste en dessous l'auraient été (0,46 % des pages gardées avant dédoublonnage)
SEUIL_DENSITE = 5
DISTINCTS_MIN = 4
NB_PROCESSUS = 4

def dossier_morceaux():
    return os.path.join(chemins.MARKETING, 'web_morceaux')

# Un processus lit un fichier entier et écrit ses pages candidates (sujet + qualité) avec leur signature.
# Le fichier « .fini » n'apparaît qu'à la fin : un fichier coupé en route est relu en entier
def traiter_fichier(fichier):
    nom = os.path.join(dossier_morceaux(), fichier.rsplit('/', 1)[-1].replace('.parquet', ''))
    compteurs = collections.Counter()
    debut = time.time()
    with open(nom + '.jsonl', 'w', encoding='utf-8') as sortie, open(nom + '.bin', 'wb') as signatures:
        for exemple in lire(fichier):
            compteurs['lues'] += 1
            normal, densite, distincts, defaut = examiner(exemple)
            if normal is None or densite < SEUIL_DENSITE or distincts < DISTINCTS_MIN:
                compteurs['hors sujet'] += 1
            elif defaut:
                compteurs[defaut] += 1
            else:
                mots_ = texte_normalise(normal).split()
                signatures.write(signature(mots_).tobytes())
                sortie.write(json.dumps({'source': 'web', 'id': exemple['id'],
                                         'titre': titre_de(exemple['text'], exemple['url']),
                                         'url': exemple['url'], 'date': exemple['date'], 'densite': round(densite, 2),
                                         'empreinte': hashlib.sha1(' '.join(mots_).encode()).hexdigest(),
                                         'texte': exemple['text']}, ensure_ascii=False) + '\n')
                compteurs['candidates'] += 1
            if compteurs['lues'] % 500_000 == 0:
                print(f"    {fichier.rsplit('/', 1)[-1]} : {compteurs['lues']:,} pages lues, {compteurs['candidates']:,} "
                      f"candidates ({compteurs['lues'] / (time.time() - debut):.0f} pages/s)", flush=True)
    with open(nom + '.fini', 'w') as f:
        json.dump(dict(compteurs), f)
    return fichier

def passe_complete():
    from multiprocessing import Pool
    os.makedirs(dossier_morceaux(), exist_ok=True)
    etat_fichier = os.path.join(chemins.MARKETING, 'web_etat.json')
    sig_fichier = os.path.join(chemins.MARKETING, 'web_signatures.bin')
    etat = json.load(open(etat_fichier)) if os.path.exists(etat_fichier) else \
        {'fusionnes': [], 'ecrites': 0, 'octets': 0, 'compteurs': {}}
    # Ce qui a été écrit après le dernier état enregistré est coupé : le morceau sera refusionné
    for chemin, taille in ((chemins.MARKETING_WEB, etat['octets']), (sig_fichier, etat['ecrites'] * NB_HACHAGES * 8)):
        if os.path.exists(chemin):
            with open(chemin, 'r+b') as f:
                f.truncate(taille)
    doublons = Doublons()
    if etat['ecrites']:
        anciennes = np.fromfile(sig_fichier, dtype=np.uint64).reshape(-1, NB_HACHAGES)
        with open(chemins.MARKETING_WEB, 'r', encoding='utf-8') as f:
            for ligne, sig in zip(f, anciennes):
                doublons.ajouter(json.loads(ligne)['empreinte'], sig)
        print(f"Reprise : {etat['ecrites']:,} pages déjà gardées, {len(etat['fusionnes'])} fichiers déjà fusionnés")
    compteurs = collections.Counter(etat['compteurs'])
    debut = time.time()

    def fusionner(fichier):
        nom = os.path.join(dossier_morceaux(), fichier.rsplit('/', 1)[-1].replace('.parquet', ''))
        compteurs.update(json.load(open(nom + '.fini')))
        candidates = np.fromfile(nom + '.bin', dtype=np.uint64).reshape(-1, NB_HACHAGES)
        with open(nom + '.jsonl', 'r', encoding='utf-8') as f, open(chemins.MARKETING_WEB, 'ab') as sortie, \
             open(sig_fichier, 'ab') as signatures:
            for ligne, sig in zip(f, candidates):
                if len(doublons.signatures) >= PLAFOND:
                    break
                empreinte = json.loads(ligne)['empreinte']
                doublon = doublons.est_doublon(empreinte, sig)
                if doublon:
                    compteurs[doublon] += 1
                    continue
                doublons.ajouter(empreinte, sig)
                signatures.write(sig.tobytes())
                sortie.write(ligne.encode('utf-8'))
            octets = sortie.tell()
        etat.update(fusionnes=etat['fusionnes'] + [fichier], ecrites=len(doublons.signatures), octets=octets,
                    compteurs=dict(compteurs))
        with open(etat_fichier + '.tmp', 'w') as f:
            json.dump(etat, f)
        os.replace(etat_fichier + '.tmp', etat_fichier)
        for extension in ('.jsonl', '.bin', '.fini'):
            os.remove(nom + extension)
        gardees = len(doublons.signatures)
        print(f"  {fichier.rsplit('/', 1)[-1]} fusionné : {gardees:,} pages gardées en tout, "
              f"{compteurs['lues']:,} lues ({gardees / max(compteurs['lues'], 1):.2%}), "
              f"{compteurs['doublon exact'] + compteurs['quasi-doublon']:,} doublons, "
              f"{(time.time() - debut) / 60:.0f} min", flush=True)

    a_faire = []
    for fichier in fichiers_du_jeu():
        if fichier in etat['fusionnes']:
            continue
        nom = os.path.join(dossier_morceaux(), fichier.rsplit('/', 1)[-1].replace('.parquet', ''))
        if os.path.exists(nom + '.fini'):
            fusionner(fichier)
        else:
            a_faire.append(fichier)
        if len(doublons.signatures) >= PLAFOND:
            break
    if len(doublons.signatures) < PLAFOND:
        with Pool(NB_PROCESSUS) as pool:
            for fichier in pool.imap_unordered(traiter_fichier, a_faire):
                fusionner(fichier)
                if len(doublons.signatures) >= PLAFOND:
                    pool.terminate()
                    break
    print(f"Terminé : {len(doublons.signatures):,} pages -> {chemins.MARKETING_WEB} "
          f"({os.path.getsize(chemins.MARKETING_WEB) / 1e6:.0f} Mo) en {(time.time() - debut) / 60:.0f} min. "
          f"Rejets : {dict(compteurs)}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--echantillon', type=int, help="note N pages pour régler le seuil")
    parser.add_argument('--montrer', type=float, help="montre les pages gardées et rejetées de justesse à ce seuil")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    os.makedirs(chemins.MARKETING, exist_ok=True)
    if args.echantillon:
        echantillon(args.echantillon)
    elif args.montrer:
        gardees, justesse = montrer(args.montrer, DISTINCTS_MIN)
        for titre, pages in (("GARDÉES", gardees), ("REJETÉES DE JUSTESSE", justesse)):
            print(f"\n===== {titre} =====")
            for p in pages:
                print(f"\n--- densité {p['densite']}, {p['distincts']} mots distincts, {p['url']}\n{p['texte'][:600]}")
    else:
        assert SEUIL_DENSITE is not None, "règle SEUIL_DENSITE après avoir regardé l'échantillon"
        passe_complete()

if __name__ == '__main__':
    main()
