import requests, re, time

# Numéros = Identifiants Project Gutenberg des livres à télécharger.
# Le nom des livres est là que pour l'affichage pendant le téléchargement. (Pas nécessaire de le mettre)
LIVRES = {
    17489:"Miserables_1", 17493:"Miserables_2", 17494:"Miserables_3",
    17518:"Miserables_4", 17519:"Miserables_5",
    17989:"MonteCristo_1", 17990:"MonteCristo_2",
    17991:"MonteCristo_3", 17992:"MonteCristo_4",
    13951:"TroisMousquetaires", 14287:"IleMysterieuse",
    5097:"VingtMilleLieues", 4791:"VoyageCentreTerre",
    20973:"TourDuMonde80", 19657:"NotreDameDeParis",
    5711:"Germinal", 5250:"Nana", 14155:"MadameBovary",
    798:"RougeEtNoir", 62215:"FantomeOpera",
    32854:"ArseneLupin", 2650:"CoteDeChezSwann",
}

def nettoyer(txt):
    # Les fichiers Gutenberg entourent le roman de deux marqueurs
    # "*** START OF ... ***" et "*** END OF ... ***" qui délimitent l'en-tête et la licence. 
    # On coupe avant et après pour ne garder que le texte du livre.
    m = re.search(r"\*\*\*\s*START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*", txt, re.S|re.I)
    if m: txt = txt[m.end():]
    m = re.search(r"\*\*\*\s*END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*", txt, re.S|re.I)
    if m: txt = txt[:m.start()]
    return txt.strip()

# Téléchargement livre par livre. Un livre qui échoue au téléchargement (réseau coupé, page déplacée),
# ne doit pas interrompre la boucle : on signale et on passe au suivant.
morceaux = []
for num, nom in LIVRES.items():
    try:
        r = requests.get(f"https://www.gutenberg.org/ebooks/{num}.txt.utf-8", timeout=60)
        r.encoding = "utf-8"  # sans ça requests devine l'encodage et casse les accents
        texte = nettoyer(r.text)
        morceaux.append(texte)
        print(f"OK  {nom:22} {len(texte):>9,} caractères")
    except Exception as e:
        print(f"ÉCHEC {nom}: {e}")
    time.sleep(0.5)  # pause entre deux requêtes pour ne pas surcharger le serveur

# Tous les livres sont collés dans un seul fichier, séparés par une ligne vide.
# C'est ce fichier que prepare.py transformera en données d'entraînement.
corpus = "\n\n".join(morceaux)
with open("corpus_fr.txt", "w", encoding="utf-8") as f:
    f.write(corpus)

print(f"\nTotal : {len(corpus):,} caractères, {len(corpus)/1_000_000:.1f} Mo")
