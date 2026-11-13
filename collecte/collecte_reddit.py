import os
import time
import praw
from collecte_commun import FICHIER_SORTIE, USER_AGENT, deja_collectes, ecrire

SOURCE = 'reddit'
SUBREDDIT = 'france'
NB_PUBLICATIONS = 1000
PERIODE = 'year'
DELAI = 0.5

identifiant = os.environ.get('REDDIT_CLIENT_ID')
secret = os.environ.get('REDDIT_CLIENT_SECRET')
if not identifiant or not secret:
    raise SystemExit("Définissez REDDIT_CLIENT_ID et REDDIT_CLIENT_SECRET "
                     "(application de type script sur reddit.com/prefs/apps)")

reddit = praw.Reddit(client_id=identifiant, client_secret=secret, user_agent=USER_AGENT)
reddit.read_only = True

SUPPRIMES = {'[deleted]', '[removed]', '[supprimé]'}

deja = deja_collectes(SOURCE)
print(f"{len(deja):,} publications déjà collectées, elles seront ignorées")

ecrits = 0
with open(FICHIER_SORTIE, 'a', encoding='utf-8') as sortie:
    try:
        publications = reddit.subreddit(SUBREDDIT).top(time_filter=PERIODE, limit=NB_PUBLICATIONS)
    except Exception as e:
        raise SystemExit(f"Connexion à Reddit impossible : {e}")

    for publication in publications:
        if publication.id in deja:
            continue

        try:
            morceaux = [publication.title, publication.selftext]
            publication.comments.replace_more(limit=0)
            for commentaire in publication.comments:
                corps = (commentaire.body or '').strip()
                if corps and corps not in SUPPRIMES:
                    morceaux.append(corps)
        except Exception as e:
            print(f"ÉCHEC publication {publication.id} : {e}")
            continue

        texte = '\n\n'.join(morceau.strip() for morceau in morceaux if morceau and morceau.strip())
        url = f"https://www.reddit.com{publication.permalink}"
        if ecrire(sortie, SOURCE, publication.id, publication.title, texte, url):
            ecrits += 1
            if ecrits % 25 == 0:
                print(f"  {ecrits:,} publications écrites")
        time.sleep(DELAI)

print(f"Terminé : {ecrits:,} publications ajoutées à {FICHIER_SORTIE}")
