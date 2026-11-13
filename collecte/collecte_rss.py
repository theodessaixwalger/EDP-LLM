import time
import feedparser
import requests
import trafilatura
from collecte_commun import FICHIER_SORTIE, deja_collectes, ecrire

SOURCE = 'presse'

FLUX = [
    "https://www.francetvinfo.fr/titres.rss",
    "https://www.numerama.com/feed/",
    "https://www.futura-sciences.com/rss/actualites.xml",
    "https://www.sciencesetavenir.fr/rss.xml",
]

DELAI = 1.5
DELAI_APRES_ERREUR = 5

ENTETES = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                         'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36'}

PAGES_INUTILES = ("javascript is disabled", "activez javascript", "activer javascript",
                  "activez les notifications", "accepter les cookies", "enable javascript")

deja = deja_collectes(SOURCE)
print(f"{len(deja):,} articles déjà collectés, ils seront ignorés")

def texte_article(lien):
    reponse = requests.get(lien, headers=ENTETES, timeout=20)
    if reponse.status_code != 200:
        print(f"  HTTP {reponse.status_code} : {lien}")
        return None
    return trafilatura.extract(reponse.content, include_comments=False, include_tables=False)

ecrits = 0
with open(FICHIER_SORTIE, 'a', encoding='utf-8') as sortie:
    for adresse_flux in FLUX:
        try:
            flux = feedparser.parse(adresse_flux)
        except Exception as e:
            print(f"ÉCHEC flux {adresse_flux} : {e}")
            continue

        if not flux.entries:
            print(f"{adresse_flux} : flux vide ou illisible, ignoré")
            continue

        print(f"{adresse_flux} : {len(flux.entries)} articles annoncés")
        for entree in flux.entries:
            lien = entree.get('link')
            if not lien or lien in deja:
                continue
            deja.add(lien)

            try:
                texte = texte_article(lien)
            except requests.RequestException as e:
                print(f"  ÉCHEC réseau {lien} : {type(e).__name__}")
                time.sleep(DELAI_APRES_ERREUR)
                continue
            except Exception as e:
                print(f"  ÉCHEC extraction {lien} : {e}")
                continue

            if texte and any(marqueur in texte[:400].lower() for marqueur in PAGES_INUTILES):
                print(f"  page de blocage ignorée : {lien}")
                texte = None

            if texte and ecrire(sortie, SOURCE, lien, entree.get('title', ''), texte, lien):
                ecrits += 1
                print(f"  OK {entree.get('title', '')[:70]}")
            time.sleep(DELAI)

print(f"Terminé : {ecrits:,} articles ajoutés à {FICHIER_SORTIE}")
