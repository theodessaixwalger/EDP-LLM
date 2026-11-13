import io
import json
import re
import requests
import pyarrow.parquet as pq
import chemins

FRENCH_QA = "https://huggingface.co/datasets/CATIE-AQ/frenchQA/resolve/main/data/train-00000-of-00001.parquet"

longueur_max_passage = 1000
longueur_max_question = 200
longueur_max_phrase = 250

def nettoyer(texte):
    return re.sub(r'\s+', ' ', texte or '').strip()

# La réponse c'est la phrase entière du passage qui contient l'extrait
def phrase_contenant(texte, debut, fin):
    gauche = max(texte.rfind(s, 0, debut) for s in ('. ', '! ', '? '))
    gauche = 0 if gauche < 0 else gauche + 2
    fins = [texte.find(s, fin) for s in ('. ', '! ', '? ')]
    droite = min([f for f in fins if f >= 0], default=len(texte) - 1)
    return texte[gauche:droite + 1].strip()

print("Téléchargement de frenchQA (~35 Mo)...")
fichier = pq.ParquetFile(io.BytesIO(requests.get(FRENCH_QA, timeout=600).content))

exemples, vus, sans_reponse = [], set(), 0
for lot in fichier.iter_batches(batch_size=5000, columns=['title', 'context', 'question', 'answers']):
    for ligne in lot.to_pylist():
        passage, question = nettoyer(ligne['context']), nettoyer(ligne['question'])
        if not passage or not question or len(passage) > longueur_max_passage \
                or len(question) > longueur_max_question:
            continue
        if (passage, question) in vus:
            continue
        vus.add((passage, question))

        reponses = ligne['answers'] or {}
        textes = reponses.get('text') or []
        if not textes:
            reponse = ''
            sans_reponse += 1
        else:
            extrait = nettoyer(textes[0])
            debut = passage.find(extrait)
            if not extrait or debut < 0:
                continue
            reponse = phrase_contenant(passage, debut, debut + len(extrait))
            if len(reponse) > longueur_max_phrase:
                reponse = extrait[0].upper() + extrait[1:]
                if not reponse.endswith('.'):
                    reponse += '.'
        exemples.append({'source': ligne['title'], 'passage': passage,
                         'question': question, 'reponse': reponse})

with open(chemins.QUESTIONS_DOCUMENTS, 'w', encoding='utf-8') as f:
    for exemple in exemples:
        f.write(json.dumps(exemple, ensure_ascii=False) + '\n')

print(f"{len(exemples):,} questions, dont {sans_reponse:,} sans réponse dans leur passage -> {chemins.QUESTIONS_DOCUMENTS}")
