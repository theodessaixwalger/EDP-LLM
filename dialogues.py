import io
import json
import re
import requests
import pyarrow.parquet as pq
import chemins
from tokeniseur import Tokeniseur

OASST = "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/data/train-00000-of-00001-b42a775f407cee45.parquet"
FRENCH_INSTRUCT = "https://huggingface.co/datasets/angeluriot/french_instruct/resolve/main/dataset.parquet"

longueur_max_question = 300
longueur_max_reponse = 400

vocab = Tokeniseur.charger().alphabet

def nettoyer(txt):
    return re.sub(r'\s*\n+\s*', ' ', txt).strip()

paires = []
def ajouter(question, reponse):
    question, reponse = nettoyer(question), nettoyer(reponse)
    if not question or not reponse:
        return
    if len(question) > longueur_max_question or len(reponse) > longueur_max_reponse:
        return
    if set(question + reponse) - vocab:
        return
    paires.append({'q': question, 'a': reponse})

# --- OpenAssistant (licence Apache 2.0) ---
print("Téléchargement d'OpenAssistant (~40 Mo)...")
table = pq.read_table(io.BytesIO(requests.get(OASST, timeout=120).content)).to_pydict()

par_id = {table['message_id'][i]: i for i in range(len(table['message_id']))}
for i in range(len(table['message_id'])):
    if table['role'][i] != 'assistant' or table['lang'][i] != 'fr':
        continue
    parent = par_id.get(table['parent_id'][i])
    if parent is None or table['role'][parent] != 'prompter' or table['lang'][parent] != 'fr':
        continue
    if table['review_result'][i]:
        ajouter(table['text'][parent], table['text'][i])
print(f"  {len(paires):,} échanges retenus")

# --- French Instruct (licence MIT) ---
print("Téléchargement de French Instruct (~180 Mo)...")
avant = len(paires)
lignes = pq.read_table(io.BytesIO(requests.get(FRENCH_INSTRUCT, timeout=600).content)).to_pylist()
for ligne in lignes:
    # On garde que les réponses écrites par des humains, pas celles générées par une IA
    if ligne['author'] != 'human' or ligne['code']:
        continue
    tours = ligne['conversation']
    if len(tours) < 2 or tours[0]['role'] != 'user' or tours[1]['role'] != 'assistant':
        continue
    reponse = tours[1]['text']
    if 'URL_' in reponse or reponse.lower().startswith('edit'):
        continue
    question = (ligne['context'] + ' ' if ligne['context'] else '') + tours[0]['text']
    ajouter(question, reponse)
print(f"  {len(paires) - avant:,} échanges retenus")

with open(chemins.DIALOGUES, 'w', encoding='utf-8') as f:
    for paire in paires:
        f.write(json.dumps(paire, ensure_ascii=False) + '\n')

print(f"{len(paires):,} échanges au total -> {chemins.DIALOGUES}")
