import pickle
import numpy as np

# Corpus brut produit par corpus.py
with open('corpus_fr.txt', 'r', encoding='utf-8') as f:
    texte = f.read()

print(f"Longueur du corpus : {len(texte):,} caractères")

# Le modèle travaille caractère par caractère : le vocabulaire est donc
# simplement l'ensemble des caractères présents dans le corpus. Le tri rend la
# numérotation reproductible (même corpus = mêmes identifiants).
caracteres = sorted(list(set(texte)))
taille_vocab = len(caracteres)
print(f"Vocabulaire : {taille_vocab} caractères uniques")

# Le réseau ne manipule que des entiers : il faut une table dans chaque sens
# pour passer du caractère à son identifiant et inversement.
car_vers_id = {c: i for i, c in enumerate(caracteres)}
id_vers_car = {i: c for i, c in enumerate(caracteres)}

# Conversion d'une chaîne complète dans un sens puis dans l'autre
def encoder(chaine):
    return [car_vers_id[c] for c in chaine]

def decoder(liste_ids):
    return ''.join([id_vers_car[i] for i in liste_ids])

# Les 10 % de fin sont mis de côté : ils servent à vérifier que le modèle
# généralise au lieu d'apprendre le corpus par cœur.
n = len(texte)
train_texte = texte[:int(n*0.9)]
val_texte = texte[int(n*0.9):]

train_ids = encoder(train_texte)
val_ids = encoder(val_texte)
print(f"Train : {len(train_ids):,} tokens")
print(f"Val   : {len(val_ids):,} tokens")

# uint16 suffit tant que le vocabulaire fait moins de 65 536 caractères, et les
# fichiers pèsent 4 fois moins qu'avec des entiers 64 bits.
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile('train.bin')
val_ids.tofile('val.bin')

# Le vocabulaire est sauvegardé avec les données : l'entraînement et la
# génération doivent utiliser exactement la même correspondance caractère/id.
meta = {
    'taille_vocab': taille_vocab,
    'car_vers_id': car_vers_id,
    'id_vers_car': id_vers_car,
}
with open('meta.pkl', 'wb') as f:
    pickle.dump(meta, f)

print("Terminé : train.bin, val.bin, meta.pkl créés")
