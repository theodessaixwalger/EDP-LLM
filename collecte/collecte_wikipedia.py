import time
import wikipediaapi
from collecte_commun import FICHIER_SORTIE, USER_AGENT, deja_collectes, ecrire

SOURCE = 'wikipedia'

CATEGORIES = ["Science", "Histoire", "Géographie", "Littérature française",
              "Informatique", "Médecine", "Philosophie", "Musique"]

PROFONDEUR_MAX = 2
ARTICLES_MAX = 5000
DELAI = 0.2

wiki = wikipediaapi.Wikipedia(user_agent=USER_AGENT, language='fr')

deja = deja_collectes(SOURCE)
print(f"{len(deja):,} articles déjà collectés, ils seront ignorés")

file_attente = [(f"Catégorie:{nom}", 0) for nom in CATEGORIES]
categories_vues = set()
articles_vus = set()
ecrits = 0

with open(FICHIER_SORTIE, 'a', encoding='utf-8') as sortie:
    while file_attente and ecrits < ARTICLES_MAX:
        titre_categorie, profondeur = file_attente.pop(0)
        if titre_categorie in categories_vues:
            continue
        categories_vues.add(titre_categorie)

        try:
            page_categorie = wiki.page(titre_categorie)
            if not page_categorie.exists():
                continue
            membres = page_categorie.categorymembers
        except Exception as e:
            print(f"ÉCHEC catégorie {titre_categorie} : {e}")
            continue

        print(f"[niveau {profondeur}] {titre_categorie} : {len(membres)} membres")
        for titre, page in membres.items():
            if ecrits >= ARTICLES_MAX:
                break

            if page.ns == wikipediaapi.Namespace.CATEGORY:
                if profondeur < PROFONDEUR_MAX:
                    file_attente.append((titre, profondeur + 1))
                continue

            if page.ns != wikipediaapi.Namespace.MAIN or titre in articles_vus or titre in deja:
                continue
            articles_vus.add(titre)

            try:
                texte = page.text
                url = page.fullurl
            except Exception as e:
                print(f"  ÉCHEC article {titre} : {e}")
                continue

            if ecrire(sortie, SOURCE, titre, titre, texte, url):
                ecrits += 1
                if ecrits % 50 == 0:
                    print(f"  {ecrits:,} articles écrits")
            time.sleep(DELAI)

print(f"Terminé : {ecrits:,} articles ajoutés à {FICHIER_SORTIE}")
