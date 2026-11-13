import collections
import heapq
import json
import os
import pickle
import re
import sys
import time
import chemins
from recherche import reparer


TAILLE_VOCAB = 8192
COUVERTURE = 0.9999
TAILLE_ECHANTILLON = 120_000_000

# L'espace reste collée à la fin du mot et le saut de ligne est toujours un morceau tout seul,
# comme ça une conversation découpée tour par tour donne pareil que découpée d'un coup
DECOUPAGE = re.compile(r"\n|[^\W\d_]+'? ?|\d ?|[^\w\s]+ ?|[^\S\n]+")

class Tokeniseur:
    anti_boucle = 6

    def __init__(self, alphabet, fusions):
        self.alphabet = set(alphabet)
        self.morceaux = list(alphabet) + [a + b for a, b in fusions]
        self.vers_id = {m: i for i, m in enumerate(self.morceaux)}
        self.rangs = {tuple(f): r for r, f in enumerate(fusions)}
        self.fusions = [tuple(f) for f in fusions]
        self.taille_vocab = len(self.morceaux)
        self.saut_de_ligne = self.vers_id['\n']
        self.cache = {}

    def encoder_mot(self, mot):
        symboles = [c for c in mot if c in self.alphabet]
        while len(symboles) > 1:
            rang, i = min((self.rangs.get((symboles[i], symboles[i + 1]), len(self.rangs)), i)
                          for i in range(len(symboles) - 1))
            if rang == len(self.rangs):
                break
            symboles[i:i + 2] = [symboles[i] + symboles[i + 1]]
        return [self.vers_id[s] for s in symboles]

    def encoder(self, texte):
        ids = []
        for mot in DECOUPAGE.findall(texte):
            codes = self.cache.get(mot)
            if codes is None:
                if len(self.cache) > 300_000:
                    self.cache.clear()
                codes = self.cache[mot] = self.encoder_mot(mot)
            ids.extend(codes)
        return ids

    def decoder(self, ids):
        return ''.join(self.morceaux[i] for i in ids)

    def nettoyer(self, texte):
        inconnus = sorted(set(texte) - self.alphabet)
        return ''.join(c for c in texte if c in self.alphabet), inconnus

    def sauvegarder(self, fichier=chemins.TOKENISEUR):
        with open(fichier, 'w', encoding='utf-8') as f:
            json.dump({'alphabet': self.morceaux[:len(self.morceaux) - len(self.fusions)],
                       'fusions': self.fusions}, f, ensure_ascii=False)

    @classmethod
    def charger(cls, fichier=chemins.TOKENISEUR):
        with open(fichier, 'r', encoding='utf-8') as f:
            donnees = json.load(f)
        return cls(donnees['alphabet'], donnees['fusions'])

# Découpage lettre par lettre des anciens modèles
class Caracteres:
    anti_boucle = 16

    def __init__(self, meta):
        self.vers_id = meta['car_vers_id']
        self.morceaux = [meta['id_vers_car'][i] for i in range(meta['taille_vocab'])]
        self.alphabet = set(self.vers_id)
        self.taille_vocab = meta['taille_vocab']
        self.saut_de_ligne = self.vers_id['\n']

    def encoder(self, texte):
        return [self.vers_id[c] for c in texte if c in self.vers_id]

    def decoder(self, ids):
        return ''.join(self.morceaux[i] for i in ids)

    def nettoyer(self, texte):
        inconnus = sorted(set(texte) - self.alphabet)
        return ''.join(c for c in texte if c in self.alphabet), inconnus

# On reconnaît le bon découpage à la taille du vocabulaire enregistrée avec les poids
def decoupage(taille_vocab):
    if os.path.exists(chemins.TOKENISEUR):
        tokeniseur = Tokeniseur.charger()
        if tokeniseur.taille_vocab == taille_vocab:
            return tokeniseur
    if os.path.exists(chemins.META):
        with open(chemins.META, 'rb') as f:
            meta = pickle.load(f)
        if meta['taille_vocab'] == taille_vocab:
            return Caracteres(meta)
    raise ValueError(f"Aucun découpage connu pour un vocabulaire de {taille_vocab} : "
                     f"relancer tokeniseur.py ou prepare.py ?")

# BPE : on fusionne à chaque tour la paire de morceaux la plus fréquente, jusqu'à 8192 morceaux
def apprendre(textes, taille_vocab=TAILLE_VOCAB):
    debut = time.time()
    compte = collections.Counter()
    for texte in textes:
        compte.update(DECOUPAGE.findall(texte))

    caracteres = collections.Counter()
    for mot, n in compte.items():
        for c in mot:
            caracteres[c] += n
    total, cumul, alphabet = sum(caracteres.values()), 0, []
    for c, n in caracteres.most_common():
        if cumul / total >= COUVERTURE and c not in '\n ':
            continue
        alphabet.append(c)
        cumul += n
    alphabet.sort()
    print(f"{len(compte):,} mots distincts, alphabet de {len(alphabet)} caractères ({time.time() - debut:.0f}s)")

    ids_alphabet = {c: i for i, c in enumerate(alphabet)}
    mots, comptes = [], []
    for mot, n in compte.items():
        if n >= 2:
            symboles = [ids_alphabet[c] for c in mot if c in ids_alphabet]
            if symboles:
                mots.append(symboles)
                comptes.append(n)
    del compte

    paires = collections.defaultdict(int)
    index = collections.defaultdict(set)
    for w, symboles in enumerate(mots):
        for p in zip(symboles, symboles[1:]):
            paires[p] += comptes[w]
            index[p].add(w)
    tas = [(-n, p) for p, n in paires.items()]
    heapq.heapify(tas)

    textes_morceaux = list(alphabet)
    fusions = []
    while len(textes_morceaux) < taille_vocab and tas:
        n, meilleure = heapq.heappop(tas)
        if paires.get(meilleure, 0) != -n or n == 0:
            continue
        nouveau = len(textes_morceaux)
        a, b = meilleure
        textes_morceaux.append(textes_morceaux[a] + textes_morceaux[b])
        fusions.append((textes_morceaux[a], textes_morceaux[b]))

        modifiees = set()
        for w in index.pop(meilleure):
            symboles = mots[w]
            n_mot = comptes[w]
            resultat, i = [], 0
            while i < len(symboles):
                if i < len(symboles) - 1 and symboles[i] == a and symboles[i + 1] == b:
                    resultat.append(nouveau)
                    i += 2
                else:
                    resultat.append(symboles[i])
                    i += 1
            if len(resultat) == len(symboles):
                continue
            for p in zip(symboles, symboles[1:]):
                paires[p] -= n_mot
                modifiees.add(p)
            for p in zip(resultat, resultat[1:]):
                paires[p] += n_mot
                index[p].add(w)
                modifiees.add(p)
            mots[w] = resultat
        paires.pop(meilleure, None)
        for p in modifiees:
            if paires.get(p, 0) > 0:
                heapq.heappush(tas, (-paires[p], p))

        if len(fusions) % 1000 == 0:
            print(f"  {len(fusions):,} fusions, dernière : {textes_morceaux[-1]!r} ({time.time() - debut:.0f}s)")
    return Tokeniseur(alphabet, fusions)

def echantillon():
    with open(chemins.CORPUS_ROMANS, 'r', encoding='utf-8') as f:
        romans = f.read()
    yield romans
    lus = len(romans)
    with open(chemins.CORPUS_COLLECTE, 'r', encoding='utf-8') as f:
        for numero, ligne in enumerate(f):
            if numero % 40:
                continue
            try:
                texte = reparer(json.loads(ligne)['texte'])
            except (json.JSONDecodeError, KeyError):
                continue
            yield texte
            lus += len(texte)
            if lus >= TAILLE_ECHANTILLON:
                break

if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    if len(sys.argv) == 1:
        tokeniseur = apprendre(echantillon())
        tokeniseur.sauvegarder()
        print(f"Vocabulaire de {tokeniseur.taille_vocab} morceaux -> {chemins.TOKENISEUR}")
    else:
        tokeniseur = Tokeniseur.charger()
        texte = ' '.join(sys.argv[1:])
        ids = tokeniseur.encoder(texte)
        print(f"{len(texte)} caractères -> {len(ids)} morceaux")
        print(' | '.join(tokeniseur.morceaux[i] for i in ids))
