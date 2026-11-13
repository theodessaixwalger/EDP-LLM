# EDP-LLM

Création d'un LLM de type GPT, écrit de zéro en PyTorch (découpage en sous-mots compris), entraîné sur des romans français classiques et toute la Wikipédia francophone, puis transformé en chatbot qui retient ce qu'on lui dit et lit Wikipédia pour répondre.

Le projet avance en trois étapes, plus une interface web :

1. **Modèle de langage** — il apprend le français et sait continuer une phrase.
2. **Chatbot avec mémoire** — réentraîné sur des conversations, il répond à tour de rôle et se souvient de ce qu'on lui a dit.
3. **Apprentissage par les préférences (DPO)** — on lui montre les bonnes et les mauvaises réponses pour corriger ses travers.

Avant chaque question de culture générale, un moteur de recherche, lui aussi écrit de zéro, place dans la conversation les articles de Wikipédia qui en parlent : le modèle y lit sa réponse. Une calculatrice et une horloge font de même pour ce qu'il ne peut pas savoir (voir [Lire Wikipédia pour répondre](#lire-wikipédia-pour-répondre)).

## Installation

```bash
python -m venv llm-env
llm-env\Scripts\activate        # Windows — source llm-env/bin/activate sous Linux/macOS
pip install -r requirements.txt
```

Pour utiliser le GPU, installez PyTorch avec la version correspondant à votre CUDA ([pytorch.org](https://pytorch.org/get-started/locally/)), par exemple `pip install torch --index-url https://download.pytorch.org/whl/cu121`.

## Organisation des fichiers

```
EDP-LLM/
├── chemins.py            tous les chemins du projet, en un seul endroit
├── model.py              l'architecture, partagée par tous les scripts
│
│   Étape 1 — modèle de langage
├── corpus.py             télécharge les romans
├── tokeniseur.py         apprend le découpage en sous-mots
├── prepare.py            découpe le corpus en morceaux, train / validation
├── train.py              entraîne le modèle
├── generate.py           complète une phrase
│
│   Étape 2 — chatbot avec mémoire
├── dialogues.py          télécharge de vraies conversations (OpenAssistant, French Instruct)
├── documents.py          télécharge des questions posées sur des textes (frenchQA)
├── recherche.py          moteur de recherche dans Wikipédia
├── outils.py             calculatrice et horloge
├── prepare_chat.py       construit les conversations d'entraînement
├── finetune.py           réentraîne le modèle en chatbot
├── chat.py               discuter dans le terminal
│
│   Étape 3 — préférences
├── preferences.py        annoter les réponses à la main
├── preferences_auto.py   générer des paires sur les erreurs connues
├── dpo.py                entraînement par préférences
├── evaluer_chat.py       mesurer le chatbot
│
│   Interface web
├── serveur.py            héberge le modèle
├── interface.html        la page de discussion
│
├── collecte/             corpus plus large : Wikipédia, presse, Reddit
├── donnees/              corpus, données d'entraînement, vocabulaire, préférences
└── modeles/              les poids appris
```

Les scripts se lancent **depuis la racine du projet**, y compris ceux de `collecte/` (`python collecte/collecte_rss.py`). Aucun nom de fichier n'est écrit en dur : tout passe par [chemins.py](chemins.py), le seul fichier à ouvrir pour savoir ce qui produit quoi.

`donnees/` et `modeles/` (environ 14 Go) ne sont pas versionnés : tout s'y régénère en relançant le pipeline. Seul `donnees/preferences.jsonl` est suivi par git, parce que des annotations faites à la main ne se régénèrent pas.

## Étape 1 — Modèle de langage

```bash
python corpus.py                          # télécharge 22 volumes de romans (17 Mo)
python collecte/collecte_huggingface.py   # Wikipédia francophone complète (6,6 Go)
python tokeniseur.py                      # apprend le découpage en sous-mots (1 min)
python prepare.py                         # découpe le corpus (5 min)
python train.py                           # entraîne (80 000 itérations, ~15 h sur RTX 3060)
python generate.py                        # tapez un début de phrase, il écrit la suite
```

Un entraînement interrompu (coupure de courant, redémarrage) reprend où il en était avec `python train.py --reprendre` : tout son état est enregistré à chaque évaluation.

`prepare.py` lit le corpus en flux, sur 4 processus, et nettoie au passage ce que le Wikipédia de Hugging Face a perdu en route : ses modèles de page ont disparu avec les dates et les nombres qu'ils affichaient, laissant « né le à Ajaccio » ou « LItalie ( ) est ». Les restes sont corrigés (« né à Ajaccio », « L'Italie est ») ; les dates, elles, sont perdues.

### Le découpage en sous-mots

Jusqu'ici le modèle lisait caractère par caractère. [tokeniseur.py](tokeniseur.py) apprend à la place, sur un échantillon du corpus, les 8 192 morceaux les plus utiles (méthode BPE, écrite de zéro) : `Utilisateur : Quelle est la capitale de la France ?` devient 14 morceaux au lieu de 51 caractères (`U|ti|lis|ateur␣|:␣|Qu|elle␣|est␣|la␣|capitale␣|de␣|la␣|France␣|?`, où `␣` est une espace). Sur tout le corpus, un morceau vaut 3,35 caractères.

- **La même fenêtre voit 3,35 fois plus de texte.** 4096 positions couvrent environ 13 700 caractères au lieu de 4096 : plus de conversation en mémoire, et des documents entiers.
- **Le modèle n'a plus à épeler.** Un mot courant est un seul morceau : ce qu'il apprend porte sur les mots et leur sens, plus sur leur orthographe.
- **L'espace est collée à la fin du mot**, et le saut de ligne est toujours un morceau à lui seul : `Assistant : ` finit donc toujours un morceau, et découper une conversation tour par tour donne exactement le même résultat que la découper d'un bloc. C'est ce qui permet à `prepare_chat.py` de mesurer chaque échange séparément.

Les anciens modèles restent utilisables : chaque script reconnaît, à la taille du vocabulaire enregistrée avec les poids, le découpage du modèle qu'il charge.

Avec 23 fois plus de Wikipédia et le découpage en sous-mots, même réseau à 10 M de paramètres près :

| | Caractères, 50 M | Sous-mots, 60 M |
|---|---|---|
| Perte de validation, ramenée au caractère¹ | 0,97 | **0,69** |
| Modèle de base : bonne fin préférée parmi 4, sur 24 débuts de phrase (hasard : 6) | 5 | 7 |
| Chatbot : bonne réponse écrite sans aide, sur les 40 questions du QCM | 0 % | **12 à 30 %**² |
| Prénom retrouvé après 60 vraies questions (~9 400 caractères) | 0/20 (hors fenêtre) | **19/20** |
| Prénom retrouvé après 150 échanges de bavardage | 0 % (hors fenêtre) | **88 à 100 %** |

¹ Sur des textes de validation différents : un ordre de grandeur, pas une mesure exacte. ² Trois fine-tunings du même modèle de base ont donné entre 18 et 30 % avant DPO, entre 12 et 25 % après. Avec 40 questions et une seule réponse tirée par question, deux mesures du même modèle peuvent déjà s'écarter de cet ordre.

Le modèle commence donc à savoir des choses par lui-même, mais peu. La mémoire, elle, a changé d'échelle : 150 échanges de bavardage tiennent encore dans la fenêtre, contre 55 avant.

À quoi s'attendre : un français correct sur plusieurs phrases, dans le style de Wikipédia, mais un sens qui ne suit pas, et très peu de culture générale (voir [Vers des réponses plus intelligentes](#vers-des-réponses-plus-intelligentes)) :

```
La capitale de l'Allemagne est → située à l'ouest de la ville de Düsseldorf
Le plus grand océan du monde est l'océan → Pacifique qui est le plus grand océan du monde
```

### Les autres collectes (optionnel)

```bash
python collecte/collecte_rss.py           # presse en ligne via flux RSS
python collecte/collecte_wikipedia.py     # Wikipédia par catégories (API MediaWiki)
python collecte/collecte_reddit.py        # r/france, nécessite REDDIT_CLIENT_ID et REDDIT_CLIENT_SECRET
```

Comme celle de Wikipédia, elles écrivent au fil de l'eau dans `donnees/corpus_brut.jsonl`, reprennent là où elles s'étaient arrêtées si on les relance, espacent leurs requêtes et ignorent les pages en erreur. Avant une grosse collecte, mettez votre adresse de contact dans le user-agent de [collecte/collecte_commun.py](collecte/collecte_commun.py).

Le Monde, Libération et Le Figaro sont exclus des flux : ils renvoient une page anti-robot qu'on collecterait à la place de l'article. OSCAR est disponible dans le script Hugging Face, mais son accès se demande manuellement.

## Étape 2 — Chatbot avec mémoire

```bash
python dialogues.py       # télécharge de vraies conversations en français
python documents.py       # télécharge des questions posées sur des textes
python recherche.py       # indexe Wikipédia pour la recherche (~4 min)
python prepare_chat.py    # construit 44 000 conversations d'entraînement (3 min)
python finetune.py        # réentraîne le modèle de l'étape 1 (~50 min)
python chat.py            # discuter
```

Dans `chat.py` : `/memoire` affiche ce que le modèle voit vraiment, `/mauvais` corrige sa dernière réponse (voir étape 3), `/oublier` efface la mémoire, entrée vide pour quitter.

### Comment il se souvient

Un LLM n'a pas d'autre mémoire que le texte qu'on lui donne. À chaque message, on lui renvoie donc la conversation dans un format fixe, et il écrit la suite jusqu'au saut de ligne :

```
Utilisateur : Je m'appelle Théo.
Assistant : Enchanté, Théo !
Utilisateur : Comment je m'appelle ?
Assistant :
```

Trois choses changent par rapport à l'étape 1 :

- **Des conversations d'entraînement** qui mêlent dialogues de romans, vraies conversations d'assistant, et faits personnels (prénom, ville, métier…) donnés puis redemandés plus loin. La réponse n'existe que plus haut dans la conversation : le modèle doit aller la chercher.
- **Une perte calculée seulement sur les réponses de l'assistant** : ce que tape l'utilisateur est imprévisible, apprendre à le deviner n'apporterait que du bruit.
- **Une fenêtre de 4096 positions** : 4096 caractères au début du projet (une cinquantaine d'échanges, voir ci-dessous), 4096 morceaux de mots aujourd'hui (plus de 150 échanges).

La conversation est sauvegardée dans `donnees/memoire.json`, mais le modèle n'en voit que les derniers échanges qui tiennent dans sa fenêtre.

### Une mémoire de 4096 caractères

La fenêtre est passée de 256 à 1024, puis à 4096 caractères. Chaque étape a buté sur un obstacle différent, chacun mesuré avant d'être corrigé :

1. **La mémoire de la carte graphique.** L'attention écrite à la main stockait une matrice 4096 × 4096 par tête : impossible au-delà de 1024. Les six têtes sont maintenant calculées en un seul appel à `scaled_dot_product_attention`, qui ne stocke jamais cette matrice, et en demi-précision (bfloat16). Résultat : 4× plus rapide, 4× moins de mémoire.
2. **Les positions apprises.** Avec une table d'un vecteur par position, un prénom donné en fin de fenêtre était oublié même 370 caractères plus loin (1/20), alors qu'en début de fenêtre il était retrouvé à 1 700 caractères (18/20). Ce n'était pas la distance qui posait problème, mais la position absolue : les vecteurs des dernières positions, peu vus à l'entraînement, étaient mal appris. Remplacés par **RoPE**, une rotation qui ne dépend que de l'écart entre deux caractères — 20/20 quelle que soit la position.
3. **La sélection du modèle.** La perte de validation globale, dominée par des répliques de roman imprévisibles, remontait dès l'itération 250 ; le modèle retenu n'avait presque rien appris. `finetune.py` choisit maintenant le modèle sur une **perte de mémoire**, calculée uniquement sur les réponses qui exigent de retrouver un fait.
4. **Les données.** Dans les conversations ordinaires, un fait est souvent redonné avant d'être redemandé : 3,7 % seulement des rappels se faisaient à plus de 2 000 caractères, et le modèle échouait exactement au-delà. 40 % des conversations sont maintenant **à longue portée** — faits donnés au début, jamais répétés, redemandés à la fin.

| Prénom redemandé… | Fenêtre 1024 | 4096, positions apprises | 4096, RoPE |
|---|---|---|---|
| après 20 / 40 / 55 échanges de bavardage | 0 / 0 / 0 % | 68 / 60 / 18 % | **100 / 90 / 75 %** |
| après 20 vraies questions (~3 100 car.) | 0/20 | 4/20 | **20/20** |
| 3 faits donnés au début, redemandés ~3 400 car. plus loin | 0/15 | ≤ 5/15 | **15, 14 et 15/15** |

Au-delà de la fenêtre, un fait est perdu, comme prévu — mais le modèle ne l'avoue pas toujours : il lui arrive de produire un prénom inventé, voire une suite de caractères répétés.

Le découpage en sous-mots a ensuite fait tenir 3,35 fois plus de texte dans les mêmes 4096 positions : le prénom est retrouvé après 60 vraies questions (19/20) et après 150 échanges de bavardage (88 à 100 %), là où le modèle caractère par caractère les avait perdus (voir [étape 1](#le-découpage-en-sous-mots)).

## Étape 3 — Apprentissage par les préférences (DPO)

Jusqu'ici, le modèle imite des textes : on lui montre la bonne réponse, jamais les mauvaises, et il répète ses travers. DPO lui présente des paires — une réponse à préférer, une à éviter.

```bash
python evaluer_chat.py                             # mesure de départ
python preferences_auto.py                         # génère des paires automatiquement
python preferences.py                              # (optionnel) annoter à la main
python dpo.py                                      # entraînement (< 1 min)
python evaluer_chat.py modeles/checkpoint_dpo.pt   # mesure d'arrivée
```

`chat.py` et l'interface web utilisent le modèle DPO dès qu'il existe. Deux garde-fous empêchent l'entraînement de tout casser : une copie figée du modèle de départ sert de référence, et un terme d'imitation (`poids_sft`) l'oblige à savoir écrire la bonne réponse.

| Mesure (modèle en sous-mots, 60 M) | Avant DPO | Après DPO |
|---|---|---|
| Rappel du prénom | 100 % | 100 % |
| Rappel de la ville | 100 % | 100 % |
| Avoue ne pas savoir | 100 % | 100 % |
| Bavardage non répété | 52 % | **100 %** |
| Question ouverte lisible | **98 %** | 82 % |
| Prénom retrouvé après 2 / 5 / 8 échanges | 100 / 100 / 100 % | 100 / 100 / 100 % |
| Prénom retrouvé après 20 / 40 / 55 / 150 échanges | 100 / 100 / 100 / 90 % | 98 / 100 / 100 / 88 % |
| **Moyenne** (hors 20 à 150) | **94 %** | **98 %** |

Ces mesures portent sur ce que les préférences visaient : elles sont flatteuses par construction. Le DPO corrige surtout le bavardage (le modèle resservait « Très bien, merci. Et vous ? » une fois sur deux), au prix de questions ouvertes un peu moins lisibles.

### Ce qu'on a appris en route

- **Il faut aussi des paires quand le modèle a raison.** Ne garder que ses erreurs déséquilibre les données : en corrigeant le rappel du prénom, une première version a fait tomber celui de la ville de 95 % à 48 %. `preferences_auto.py` fabrique donc aussi une mauvaise réponse quand il répond juste.
- **Un bon score DPO ne garantit pas un bon modèle.** Deux essais ont atteint 97 % d'accord sur les préférences de validation tout en produisant un chatbot qui refusait tout (68 % de moyenne). D'où `evaluer_chat.py`, qui mesure des comportements et non une perte.
- **Préférer sa propre bonne réponse.** Quand le modèle répond juste, la réponse à préférer est la sienne, pas un modèle de phrase. Imposer « Vous habitez à Lyon » à un modèle qui disait « Vous vivez à Lyon » ne corrige rien mais déplace tout son style : en conversation, il s'est mis à répondre « Vous habitez Nathan », et la ville retrouvée est tombée de 13 à 2 fois sur 15. Avec sa propre réponse, 10 fois sur 15.
- **Chercher les erreurs du bon modèle.** Les paires automatiques venaient du DPO précédent, qui ne se trompait presque plus : 2 erreurs réelles sur 600 situations. Elles viennent maintenant du modèle de l'étape 2, celui que le DPO va affiner : 44 erreurs.
- **Une tournure peut faire bégayer.** En sous-mots, préférer « Vous vous appelez Théo. » renforçait « vous » après « Vous » : le modèle s'est mis à écrire « Vous vous vous vous préférez le Théo. », 6 fois sur 36. Avec « Votre prénom est Théo. », l'autre tournure apprise au fine-tuning, une seule fois.
- **Une règle trop fine pour un petit modèle.** « Répondre si l'information est dans la conversation, l'avouer sinon » ne s'est pas appris à 11 M de paramètres : le modèle s'effondrait sur un raccourci. Le contournement : l'âge sert seul à apprendre l'aveu d'ignorance, et les autres faits ne sont jamais demandés sans avoir été donnés.

**Limite connue** : l'âge n'est donc jamais rappelé. Dites-lui « j'ai 34 ans » puis « quel âge j'ai », il répondra qu'il ne le sait pas. La variante qui corrige ce point le fait inventer des âges jamais donnés, ce qui est pire qu'avouer.

À surveiller si vous enchaînez les tours de DPO : le modèle répond « Parfait. » de plus en plus souvent. L'annotation manuelle est le remède.

## Vers des réponses plus intelligentes

La mémoire réglée, le modèle répondait toujours à côté : « Quelle est la capitale de la France ? » → « La capitale de la France est la capitale de la France. » Trois essais, chacun mesuré :

1. **Des réponses écrites par des humains.** `dialogues.py` ajoute 28 149 questions-réponses de French Instruct, en écartant celles générées par une IA. Les réponses restent dans le sujet, mais elles sont fausses : au QCM de culture générale d'`evaluer_chat.py` (40 questions à 4 choix), le modèle reste au niveau du hasard.
2. **Un anti-boucle.** Un petit modèle s'enferme vite dans « de plus de plus de plus ». À la génération, `bloquer_repetitions` ([model.py](model.py)) interdit de recopier une suite de 16 caractères déjà écrite dans la réponse : 2 réponses en boucle sur 24 au lieu de 11.
3. **Un modèle 4,5 fois plus gros** : 50 M de paramètres au lieu de 11 M. La perte de validation de l'étape 1 passe de 1,05 à 0,97, mais le modèle de base, testé avant toute conversation sur 24 débuts de phrase (« La capitale de l'Italie est… »), ne choisit la bonne fin parmi 4 que 5 fois. C'est le hasard, comme à 11 M (6 fois), et il ne l'écrit jamais de lui-même.

**La taille ne suffit pas : ce sont les données qui manquent.** Le corpus de l'étape 1 est fait des 20 000 premiers articles de Wikipédia, dominés par les fiches de communes : « est une commune » y revient 4 906 fois, « capitale de la France » 13 fois en 282 millions de caractères. Le modèle a appris à écrire comme Wikipédia (« La capitale de la France est… le canton de Brest »), pas les faits qu'elle contient.

La mémoire lointaine, que les instructions avaient fait reculer, est en revanche revenue. La part des faits a été remontée en même temps que la taille du modèle, on ne peut donc pas dire lequel des deux changements l'a ramenée :

| Après DPO | 11 M | 11 M + instructions | 50 M + instructions |
|---|---|---|---|
| Prénom après 20 vraies questions (~3 100 car.) | 20/20 | 17/20 | **20/20** |
| 3 faits donnés au début, redemandés ~3 400 car. plus loin | 15, 14 et 15/15 | 15, 7 et 6/15 | **15, 13 et 13/15** |

D'où deux chantiers : lui donner les faits au moment où on les lui demande (ci-dessous), et l'entraîner sur toute la Wikipédia, 23 fois plus de texte, découpée en sous-mots ([étape 1](#le-découpage-en-sous-mots)).

## Lire Wikipédia pour répondre

Un modèle de 60 M de paramètres ne retiendra jamais toute Wikipédia. Il sait en revanche recopier : c'est ce qui lui fait retrouver un prénom donné 50 échanges plus haut. Avant une question de culture, on place donc dans la conversation les articles qui en parlent, et il y lit sa réponse :

```
Document : Les Misérables, également nommé Les Misérables du , est un film français réalisé par Claude Lelouch et sorti en 1995. […]
Document : Les Misérables est un roman de Victor Hugo publié en 1862, l’un des plus vastes et des plus notables de la littérature du . […]
Utilisateur : Qui a écrit Les Misérables ?
Assistant : Les Misérables est un roman de Victor Hugo publié en 1862, l’un des plus vastes et des plus notables […]
```

(Derrière les « du , » et « du . », des siècles effacés du Wikipédia de Hugging Face.)

```bash
python recherche.py "Qui a écrit Les Misérables ?"   # voir ce que la recherche trouve
```

- **La recherche** ([recherche.py](recherche.py)) : BM25, écrit de zéro, sur le début de 699 489 articles (jusqu'à 700 caractères chacun). Les mots du titre comptent triple, un titre d'article cité tel quel dans la question fait remonter l'article, et à score égal, un grand article passe devant un petit : « Rome » devant les ambassades d'Italie, « Ottawa » devant « Gatineau ». L'index (545 Mo) est lu sur le disque sans être chargé en mémoire : une recherche prend une vingtaine de millisecondes.
- **Seulement pour la culture générale.** « Comment je m'appelle », « je suis fatigué » ou « tu aimes la musique ? » trouvent leur réponse dans la conversation, pas dans Wikipédia : aucune recherche n'est faite. « Explique-moi la photosynthèse », si.
- **Seulement pour la question en cours.** Les documents des questions précédentes sont retirés de la mémoire : ils prendraient la place de la conversation.
- **Apprendre à lire.** 35 % des conversations d'entraînement finissent par une question précédée de documents : 48 092 questions de [frenchQA](https://huggingface.co/datasets/CATIE-AQ/frenchQA), dont la réponse est la phrase du texte qui la contient ([documents.py](documents.py)), et 54 890 questions de définition tirées des débuts d'articles (« C'est quoi X ? »). Une fois sur sept, le passage ne répond pas, et le modèle doit le dire (« Je ne trouve pas la réponse dans ce document. ») plutôt qu'inventer. Sept fois sur dix, un second passage trouvé par la même recherche, mais qui n'est pas le bon, sert de leurre.

Mesuré avec les 40 questions du QCM posées en discussion, après DPO :

| | Caractères, 50 M | Sous-mots, 60 M |
|---|---|---|
| Bonne réponse écrite sans documents | 0 % | 15 % |
| Bonne réponse écrite avec documents | **28 %** | **28 %** |
| Réponse présente dans les documents trouvés | 57 % | 60 % |

- **L'ordre des documents compte.** Le modèle recopie plus volontiers le document le plus proche de la question : placer le meilleur en dernier double les bonnes réponses (11 % → 22 %). À l'entraînement, le bon passage était presque toujours le dernier, faute de leurre ; il y en a maintenant un sept fois sur dix, comme en discussion où la recherche en renvoie deux.
- **La recherche fixe le plafond.** Le modèle ne peut pas lire ce qu'on ne lui donne pas. Favoriser les grands articles a fait passer la réponse apportée de 46 à 52 fois sur 71 questions (le QCM et 33 autres, écrites avant le réglage), mais « Où est né Napoléon » trouve toujours Pierre-Napoléon Bonaparte, et les dates perdues par le Wikipédia de Hugging Face ne se retrouvent nulle part.
- **Un document faux fait plus de mal qu'aucun.** Le modèle en sous-mots connaît une partie des réponses par lui-même, mais quand la recherche se trompe, il recopie le mauvais article au lieu de répondre de mémoire. C'est pour cela que l'écart avec et sans documents se resserre.

### Outils : calculatrice et horloge

Un modèle de 60 M de paramètres ne sait pas calculer, et rien dans ses données ne lui dit quel jour on est. [outils.py](outils.py) s'en charge avant lui, comme la recherche, et place son résultat en document :

```
Document : 17 × 23 = 391.
Utilisateur : Combien font 17 fois 23 ?
Assistant : 17 × 23 = 391.
```

- **La calculatrice** comprend les nombres en lettres (« quatre-vingt-dix-sept », « deux mille cinq cents »), les quatre opérations, les puissances et les pourcentages. Elle n'utilise jamais `eval()` : l'expression est analysée par `ast` et seuls les nombres et les opérations passent, ce qui rend inoffensif un message comme `__import__('os')`. Elle ne se déclenche pas sur « la guerre de 1914-1918 » ni sur « le plus grand océan ».
- **L'horloge** répond à « Quel jour sommes-nous ? » ou « Quelle heure est-il ? », mais pas à « En quelle année a eu lieu la Révolution ? » ni à « À quelle heure ouvre le Louvre ? ».
- **Apprendre à s'en servir.** Le modèle savait déjà recopier un document, mais pas parfaitement : 21 fois sur 30, avec « 12 + 84. » pour « 12 × 7 = 84. » et « Je suis le mardi 27 octobre » pour la date. Une question sur cinq parmi celles à document est donc maintenant un calcul ou une date, avec le résultat de l'outil lui-même : **30 fois sur 30** ensuite, et 90 % sur 40 calculs et dates tirés au hasard (`evaluer_chat.py`).

## Interface web

```bash
python serveur.py         # puis ouvrir http://127.0.0.1:8000
```

Une interface de discussion classique, avec trois ajouts :

- **La fenêtre de mémoire** dans le panneau latéral : ce que le modèle relit, la place qui reste, et les échanges qui en sont sortis — encore dans le fichier, mais oubliés par le modèle.
- **Les sources** : sous une réponse, les articles de Wikipédia que le modèle a lus, ou l'outil qui a fait le calcul.
- **La correction en un clic** : sous la dernière réponse, « Mauvaise réponse » propose de la faire régénérer ou d'écrire la bonne. La paire part dans `donnees/preferences.jsonl`, prête pour `dpo.py`.

Les réponses s'affichent morceau par morceau, au rythme où le modèle les produit : sur RTX 3060, environ 140 morceaux (580 caractères) par seconde en début de conversation, 22 (60 caractères) quand la fenêtre est pleine. Le serveur n'utilise que la bibliothèque standard de Python et n'écoute que sur `127.0.0.1`.

## Architecture du modèle

GPT décodeur seul, même réseau pour les trois étapes :

| | Étape 1 | Étape 2 |
|---|---|---|
| Embedding / têtes / blocs | 640 / 10 / 10 | 640 / 10 / 10 |
| Fenêtre de contexte | 4096 morceaux (~13 700 caractères) | 4096 morceaux |
| Positions | RoPE (aucun paramètre) | RoPE |
| Vocabulaire | 8 192 morceaux | 8 192 morceaux |
| Paramètres | 59,71 M | 59,71 M |
| Batch / itérations | 4 / 80 000 (1,3 milliard de morceaux, ~15 h) | 4 / 3 000 (~50 min) |
| Taux d'apprentissage / dropout | 3e-4 → 3e-5 / 0 | 1e-4 / 0,1 |
| Perte de validation | 2,31 par morceau | mémoire : 0,080 |

Les 10 M de paramètres de plus qu'au modèle caractère par caractère sont ceux des tables d'entrée et de sortie, une ligne par morceau du vocabulaire. Le dropout est à 0 à l'étape 1 : les 80 000 itérations ne font même pas un tour complet des 1,7 milliard de morceaux du corpus, le modèle n'a pas le temps d'apprendre un texte par cœur.

L'étape 2, elle, surapprend vite : une conversation de 4096 morceaux contient ~34 échanges au lieu de 11 en caractères, et les 39 000 vrais échanges y reviennent trois fois plus souvent. À 8 000 itérations, la perte de validation remontait de 1,96 à 2,17 et le modèle ne répondait juste sans Wikipédia que 18 fois sur 100, contre 30 à 3 000 itérations.

À l'étape 1, le taux d'apprentissage monte pendant 1 000 itérations puis décroît en cosinus, et la norme des gradients est plafonnée à 1 (aussi à l'étape 2). L'étape 3 reprend le réseau de l'étape 2 sans le modifier. Le modèle calcule en bfloat16 sur les cartes qui le gèrent, les poids restant stockés en float32. Seul `train.py` fixe la taille du réseau : elle est enregistrée avec les poids, et `MonLLM.charger()` reconstruit le modèle à partir du seul fichier.

## Données

| Étape | Source | Volume |
|---|---|---|
| 1 | Romans du domaine public (Project Gutenberg) : *Les Misérables*, *Le Comte de Monte-Cristo*, *Germinal*… — liste complète dans [corpus.py](corpus.py) | 17 Mo |
| 1 | Wikipédia francophone ([wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia), novembre 2023), articles de plus de 2 000 caractères | 936 987 articles, 6,6 Go |
| 1 | Presse en ligne (franceinfo, Numerama, Futura, Sciences et Avenir) | 146 articles |
| 2 | Débuts des mêmes articles de Wikipédia, pour la recherche | 699 489 passages |
| 2 | [frenchQA](https://huggingface.co/datasets/CATIE-AQ/frenchQA), questions posées sur des textes | 48 092 avec réponse, 102 060 sans |
| 2 | Dialogues extraits des romans | ~10 000 échanges |
| 2 | [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1) (licence Apache 2.0), conversations françaises validées | 750 échanges |
| 2 | [French Instruct](https://huggingface.co/datasets/angeluriot/french_instruct) (licence MIT), réponses écrites par des humains | 28 149 échanges |
| 2 | Conversations générées autour de 6 faits personnels, dont 40 % à longue portée | 44 000 conversations |

Les conversations générées utilisent des noms inventés 7 fois sur 10, pour que le modèle apprenne à recopier plutôt qu'à mémoriser, et des variantes de frappe (minuscules, ponctuation oubliée) pour qu'il reconnaisse `je m'appelle léo` autant que `Je m'appelle Léo.`.
