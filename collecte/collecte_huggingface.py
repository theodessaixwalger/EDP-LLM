from datasets import load_dataset
from collecte_commun import FICHIER_SORTIE, deja_collectes, preparer

SOURCE = 'huggingface'

JEU = 'wikimedia/wikipedia'
CONFIG = '20231101.fr'

NB_DOCUMENTS = None
# Les articles trop courts sont surtout des fiches de communes, ça apprend rien
LONGUEUR_MINIMALE_ARTICLE = 2000
TAILLE_LOT = 500

flux = load_dataset(JEU, CONFIG, split='train', streaming=True)

deja = deja_collectes(SOURCE)
print(f"{len(deja):,} documents déjà collectés, ils seront ignorés")

lot = []
ecrits = 0

with open(FICHIER_SORTIE, 'a', encoding='utf-8') as sortie:
    for exemple in flux:
        if NB_DOCUMENTS is not None and ecrits >= NB_DOCUMENTS:
            break
        if len(exemple.get('text', '')) < LONGUEUR_MINIMALE_ARTICLE:
            continue

        identifiant = exemple.get('id') or exemple.get('url') or exemple.get('title')
        if identifiant in deja:
            continue

        ligne = preparer(SOURCE, identifiant, exemple.get('title', ''),
                         exemple.get('text', ''), exemple.get('url', ''))
        if ligne:
            lot.append(ligne)

        if len(lot) >= TAILLE_LOT:
            sortie.writelines(lot)
            sortie.flush()
            ecrits += len(lot)
            lot.clear()
            if ecrits % 20000 == 0:
                print(f"  {ecrits:,} documents écrits")

    if lot:
        sortie.writelines(lot)
        sortie.flush()
        ecrits += len(lot)

print(f"Terminé : {ecrits:,} documents ajoutés à {FICHIER_SORTIE}")
