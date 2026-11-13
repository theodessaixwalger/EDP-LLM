import collections
import json
import re
import time
import zlib
from multiprocessing import Pool
import numpy as np
import chemins
from recherche import reparer
from tokeniseur import Tokeniseur

# 1 document sur 200 part en validation
PART_VAL = 200
PART_RARES_MAX = 0.01
# Les romans sont tout petits à côté de Wikipédia, on les fait lire 3 fois
REPETITIONS_ROMANS = 3
NB_PROCESSUS = 4
TAILLE_LOT = 2_000_000

FIN_ARTICLE = re.compile(r'\n(Notes et références|Références|Notes|Voir aussi|Liens externes|'
                         r'Bibliographie|Articles connexes|Annexes)\n')

def documents():
    with open(chemins.CORPUS_ROMANS, 'r', encoding='utf-8') as f:
        romans = [r.strip() for r in re.split(r'\n{3,}', f.read()) if r.strip()]
    for _ in range(REPETITIONS_ROMANS):
        yield from romans
    with open(chemins.CORPUS_COLLECTE, 'r', encoding='utf-8') as f:
        for ligne in f:
            try:
                texte = json.loads(ligne).get('texte', '')
            except json.JSONDecodeError:
                continue
            coupure = FIN_ARTICLE.search(texte)
            if coupure:
                texte = texte[:coupure.start()]
            # Le Wikipédia de Hugging Face a perdu ses dates et nombres, on répare ce qui reste (« né le à Ajaccio »)
            texte = reparer(texte)
            if texte.strip():
                yield texte.strip()

def lots():
    lot, taille = [], 0
    for texte in documents():
        lot.append(texte)
        taille += len(texte)
        if taille >= TAILLE_LOT:
            yield lot
            lot, taille = [], 0
    if lot:
        yield lot

# --- Travail d'un processus ---
tokeniseur = longueurs = None

def initialiser():
    global tokeniseur, longueurs
    tokeniseur = Tokeniseur.charger()
    longueurs = np.array([len(m) for m in tokeniseur.morceaux])

def encoder_lot(lot):
    sorties = {'train': [], 'val': []}
    ecartes = 0
    saut = [tokeniseur.saut_de_ligne] * 2
    for texte in lot:
        ids = np.array(tokeniseur.encoder(texte), dtype=np.uint16)
        # Trop de caractères hors vocabulaire (grec, chinois...) : on jette le document
        if longueurs[ids].sum() < len(texte) * (1 - PART_RARES_MAX):
            ecartes += 1
            continue
        # Choisi avec un hash du texte, comme ça un document tombe toujours du même côté
        cote = 'val' if zlib.crc32(texte[:200].encode()) % PART_VAL == 0 else 'train'
        sorties[cote].append(ids)
        sorties[cote].append(np.array(saut, dtype=np.uint16))
    octets = {c: (np.concatenate(v).tobytes() if v else b'') for c, v in sorties.items()}
    return octets['train'], octets['val'], sum(map(len, lot)), ecartes

if __name__ == '__main__':
    debut = time.time()
    totaux = collections.Counter()
    with Pool(NB_PROCESSUS, initializer=initialiser) as pool, \
         open(chemins.TRAIN_BIN, 'wb') as train, open(chemins.VAL_BIN, 'wb') as val:
        en_cours = collections.deque()

        def ecrire(resultat):
            octets_train, octets_val, nb_caracteres, ecartes = resultat
            train.write(octets_train)
            val.write(octets_val)
            totaux['train'] += len(octets_train) // 2
            totaux['val'] += len(octets_val) // 2
            totaux['caracteres'] += nb_caracteres
            totaux['ecartes'] += ecartes
            if totaux['lots'] % 200 == 0:
                print(f"  {totaux['caracteres'] / 1e9:.2f} G caractères -> {totaux['train'] / 1e6:,.0f} M morceaux "
                      f"({time.time() - debut:.0f}s)")
            totaux['lots'] += 1

        for lot in lots():
            en_cours.append(pool.apply_async(encoder_lot, (lot,)))
            if len(en_cours) >= 2 * NB_PROCESSUS:
                ecrire(en_cours.popleft().get())
        while en_cours:
            ecrire(en_cours.popleft().get())

    print(f"\n{totaux['caracteres']:,} caractères lus, {totaux['ecartes']:,} documents écartés (autre alphabet)")
    print(f"Train : {totaux['train']:,} morceaux, Val : {totaux['val']:,} morceaux "
          f"({totaux['caracteres'] / (totaux['train'] + totaux['val']):.2f} caractères par morceau)")
    print(f"Terminé en {time.time() - debut:.0f}s : {chemins.TRAIN_BIN}, {chemins.VAL_BIN}")
