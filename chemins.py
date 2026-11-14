import os

# Tous les fichiers du projet sont déclarés ici et nulle part ailleurs.
# Les chemins sont relatifs : les scripts se lancent depuis la racine du projet.

DONNEES = 'donnees'
MODELES = 'modeles'

# --- Étape 1 : corpus et modèle de langage ---
CORPUS_ROMANS = os.path.join(DONNEES, 'corpus_fr.txt')
CORPUS_COLLECTE = os.path.join(DONNEES, 'corpus_brut.jsonl')
TRAIN_BIN = os.path.join(DONNEES, 'train.bin')
VAL_BIN = os.path.join(DONNEES, 'val.bin')
META = os.path.join(DONNEES, 'meta.pkl')
TOKENISEUR = os.path.join(DONNEES, 'tokeniseur.json')
MODELE_BASE = os.path.join(MODELES, 'checkpoint.pt')

# --- Étape 2 : chatbot avec mémoire ---
DIALOGUES = os.path.join(DONNEES, 'dialogues_fr.jsonl')
CHAT_TRAIN = os.path.join(DONNEES, 'chat_train.npz')
CHAT_VAL = os.path.join(DONNEES, 'chat_val.npz')
MODELE_CHAT = os.path.join(MODELES, 'checkpoint_chat.pt')

# --- Étape 3 : apprentissage par les préférences ---
PREFERENCES = os.path.join(DONNEES, 'preferences.jsonl')
MODELE_DPO = os.path.join(MODELES, 'checkpoint_dpo.pt')

# --- Lecture de documents ---
QUESTIONS_DOCUMENTS = os.path.join(DONNEES, 'questions_documents.jsonl')
RECHERCHE = os.path.join(DONNEES, 'recherche')
PASSAGES = os.path.join(RECHERCHE, 'passages.jsonl')
LECTURE = os.path.join(DONNEES, 'lecture.jsonl')

# --- Base documentaire marketing ---
MARKETING = os.path.join(DONNEES, 'marketing')
DUMPS_WIKIPEDIA = os.path.join(MARKETING, 'dumps')
CATEGORIES_MARKETING = os.path.join(MARKETING, 'categories.json')
MARKETING_WIKIPEDIA = os.path.join(MARKETING, 'wikipedia.jsonl')
WIKIPEDIA_COMPLEMENT = os.path.join(MARKETING, 'wikipedia_complement.jsonl')
MARKETING_WEB = os.path.join(MARKETING, 'web.jsonl')
FICHES = os.path.join(MARKETING, 'fiches')
RECHERCHE_MARKETING = os.path.join(MARKETING, 'recherche')
MMLU_MARKETING = os.path.join(MARKETING, 'mmlu_marketing_fr.jsonl')

# --- Évaluation ---
QCM = os.path.join(DONNEES, 'qcm', 'qcm_v2.jsonl')

# --- Discussion ---
MEMOIRE = os.path.join(DONNEES, 'memoire.json')

# Le modèle DPO remplace celui du chat dès qu'il existe
def modele_le_plus_recent():
    return MODELE_DPO if os.path.exists(MODELE_DPO) else MODELE_CHAT
