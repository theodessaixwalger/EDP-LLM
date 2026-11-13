import json
import os
import re
import unicodedata

FICHIER_SORTIE = 'donnees/corpus_brut.jsonl'

# Mettre son adresse de contact avant une grosse collecte
USER_AGENT = "EDP-LLM/1.0 (projet etudiant; contact: votre.email@exemple.fr)"

LONGUEUR_MINIMALE = 200

INVISIBLES = re.compile(r'[\x00-\x08\x0b-\x1f\x7f-\x9f​-‏  ﻿]')

def nettoyer(texte):
    if not texte:
        return ''
    texte = unicodedata.normalize('NFC', texte)
    texte = INVISIBLES.sub('', texte)
    texte = texte.replace('\r\n', '\n').replace('\r', '\n')
    texte = re.sub(r'[ \t]+', ' ', texte)
    texte = re.sub(r' *\n *', '\n', texte)
    texte = re.sub(r'\n{3,}', '\n\n', texte)
    return texte.strip()

def deja_collectes(source):
    vus = set()
    if not os.path.exists(FICHIER_SORTIE):
        return vus
    with open(FICHIER_SORTIE, 'r', encoding='utf-8') as f:
        for ligne in f:
            try:
                document = json.loads(ligne)
            except json.JSONDecodeError:
                continue
            if document.get('source') == source:
                vus.add(document.get('id'))
    return vus

def preparer(source, identifiant, titre, texte, url=''):
    texte = nettoyer(texte)
    if len(texte) < LONGUEUR_MINIMALE:
        return None
    document = {
        'source': source,
        'id': identifiant,
        'titre': nettoyer(titre),
        'texte': texte,
        'url': url,
    }
    return json.dumps(document, ensure_ascii=False) + '\n'

def ecrire(fichier, source, identifiant, titre, texte, url=''):
    ligne = preparer(source, identifiant, titre, texte, url)
    if ligne is None:
        return False
    fichier.write(ligne)
    fichier.flush()
    return True
