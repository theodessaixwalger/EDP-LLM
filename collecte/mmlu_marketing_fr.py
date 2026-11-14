import html
import io
import json
import os
import sys

import pyarrow.parquet as pq
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chemins

# Le sous-ensemble marketing de MMLU traduit en français (neuralmagic/mmlu_fr), comme second jeu d'évaluation.
# MMLU est sous licence MIT, mais la traduction ne déclare aucune licence : on la télécharge pour évaluer
# en local et on ne la met pas dans le dépôt (donnees/marketing/ est ignoré par git)
URL = 'https://huggingface.co/datasets/neuralmagic/mmlu_fr/resolve/main/marketing.parquet'

def nettoyer(texte):
    return ' '.join(html.unescape(texte).split())

table = pq.read_table(io.BytesIO(requests.get(URL, timeout=120).content))
os.makedirs(chemins.MARKETING, exist_ok=True)
with open(chemins.MMLU_MARKETING, 'w', encoding='utf-8') as f:
    for n, ligne in enumerate(table.to_pylist()):
        f.write(json.dumps({'id': f"mmlu_{n + 1:03d}", 'question': nettoyer(ligne['question']),
                            'choix': [nettoyer(c).rstrip('.') for c in ligne['choices']],
                            'reponse': 'ABCD'.index(ligne['answer']), 'categorie': 'MMLU marketing', 'type': 'texte',
                            'source': 'mmlu', 'article': '', 'url': ''}, ensure_ascii=False) + '\n')
print(f"{table.num_rows} questions -> {chemins.MMLU_MARKETING}")
