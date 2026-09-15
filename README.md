# EDP-LLM

Création d'un LLM de type GPT (character-level), entraînée sur un corpus de littérature française classique (domaine public, Project Gutenberg).

## Structure du projet

| Fichier | Rôle |
|---|---|
| [corpus.py](corpus.py) | Télécharge ~19 romans classiques depuis Project Gutenberg et construit `corpus_fr.txt` |
| [prepare.py](prepare.py) | Tokenisation au niveau caractère, split train/val, génère `train.bin`, `val.bin`, `meta.pkl` |
| [model.py](model.py) | Architecture du modèle (attention multi-têtes, blocs transformer, `MonLLM`) |
| [train.py](train.py) | Boucle d'entraînement, sauvegarde `checkpoint.pt` |
| [generate.py](generate.py) | Charge le modèle entraîné et complète les phrases qu'on lui donne |

Les fichiers générés (`corpus_fr.txt`, `train.bin`, `val.bin`, `meta.pkl`, `checkpoint.pt`) ne sont pas versionnés (voir `.gitignore`) : ils se recréent en exécutant le pipeline.

## Installation

```bash
python -m venv llm-env
llm-env\Scripts\activate      # Windows
# source llm-env/bin/activate # Linux/macOS

pip install -r requirements.txt
```

> Pour utiliser un GPU CUDA, installez PyTorch avec la roue adaptée à votre version de CUDA via [pytorch.org](https://pytorch.org/get-started/locally/) (ex. `pip install torch --index-url https://download.pytorch.org/whl/cu121`) plutôt que la version CPU par défaut de PyPI.

## Utilisation

Exécuter les étapes dans l'ordre :

```bash
python corpus.py    # télécharge et nettoie le corpus (~17 Mo, ~19 romans)
python prepare.py   # tokenise et prépare les données d'entraînement
python train.py     # entraîne le modèle (5000 itérations)
python generate.py  # teste le modèle : on donne une phrase, il écrit la suite
```

## Tester le modèle

Une fois l'entraînement fait, il n'y a plus rien à relancer du pipeline : `generate.py` a besoin uniquement de `checkpoint.pt` et `meta.pkl`.

Le script charge le modèle une seule fois, puis attend des phrases. On tape une amorce, le modèle écrit la suite, et on peut en essayer autant qu'on veut. Entrée vide pour quitter.

```bash
python generate.py
```

Exemple de session :

```
Modèle chargé sur cuda. Entrée vide pour quitter.

Amorce > Le soir tombait sur la ville

Le soir tombait sur la ville de son maître qui s'était assuré son
arbre; Milady faisait de ses yeux.

«Si vous ne vous avez pas encore, je me serrai que j'ai le soir.

--Et vous revoyez mon cher ai-je sur mon fils? dit Morrel avec lui;
mais je me suis refart où je vous crois que vous ne voulez pas ce qui
vous ne m'a pas répondu [...]

Amorce >
```

Deux réglages en haut de [generate.py](generate.py) :

- `nb_caracteres` — longueur du texte écrit après l'amorce (500 par défaut)
- `temperature` — 0.8 par défaut. En dessous de 0.5 le texte devient propre mais répétitif ; au dessus de 1.2 le modèle invente des mots.

L'amorce doit être écrite avec des caractères vus pendant l'entraînement (lettres, accents, ponctuation classique). Un caractère absent du corpus — un emoji par exemple — est signalé puis ignoré.

Deux repères pour juger le résultat : le modèle ne connaît que 256 caractères de contexte (environ 3 lignes), et il écrit caractère par caractère sans notion de mot. Une orthographe correcte et une ponctuation cohérente sur plusieurs phrases sont donc déjà un bon signe.

## Architecture du modèle

GPT décodeur-seul, configuration par défaut :

- Dimension d'embedding : 384
- Têtes d'attention : 6
- Blocs transformer : 6
- Taille de contexte : 256 tokens
- Dropout : 0.2
- ~10-11M de paramètres
- Tokenisation caractère par caractère (pas de BPE)

## Données

Corpus constitué de romans du domaine public (Project Gutenberg) : *Les Misérables*, *Le Comte de Monte-Cristo*, *Les Trois Mousquetaires*, *Notre-Dame de Paris*, *Germinal*, *Nana*, *Madame Bovary*, *Le Rouge et le Noir*, *Le Fantôme de l'Opéra*, *Arsène Lupin*, *Du côté de chez Swann*, *L'Île mystérieuse*, *Vingt mille lieues sous les mers*, *Voyage au centre de la Terre*, *Le Tour du monde en quatre-vingts jours*, et autres — voir [corpus.py](corpus.py) pour la liste complète.
