import collections
import json
import random
import re
import sys
import numpy as np
import chemins
from evaluer_qcm import avec_les_choix, cle
from recherche import Recherche, cle_titre, normaliser

# Exemples de compréhension de lecture fabriqués par règles à partir des débuts d'articles :
# on repère un fait dans le passage (« né à Y », « fondé en Y »...), on en fait une question,
# et on choisit les distracteurs parmi les réponses du même genre trouvées dans d'autres articles
NB_EXEMPLES = 15000
LONGUEUR_MIN_ARTICLE = 30000
PARTS = {'qcm': 0.40, 'courte': 0.30, 'sans_reponse': 0.15, 'plusieurs': 0.15}
# Un QCM sur huit sans document : le format doit tenir même quand la recherche ne trouve rien
PART_QCM_SANS_DOCUMENT = 0.125
PART_DEUX_DOCUMENTS = 0.6
PART_JUSTIFICATION = 0.5
MAX_PAR_FAIT = {'region': 0.06, 'departement': 0.08, 'definition': 0.18}
N_GRAMME = 8

# Les mêmes que dans prepare_chat.py
AUCUNE_REPONSE = ["Je ne trouve pas la réponse dans ce document.",
                  "Ce document ne répond pas à la question."]

NOM = (r"([A-ZÉÈÊÀÂÎÔÛÇŒ][\w'’\-]*(?:[ \-](?:de la |de |d'|d’|du |des |la |le |les |l'|y |e )?"
       r"[A-ZÉÈÊÀÂÎÔÛÇŒ][\w'’\-]*){0,3})")
NATIONALITES = [
    ('français', 'française'), ('américain', 'américaine'), ('britannique', 'britannique'),
    ('anglais', 'anglaise'), ('allemand', 'allemande'), ('italien', 'italienne'), ('espagnol', 'espagnole'),
    ('belge', 'belge'), ('suisse', 'suisse'), ('canadien', 'canadienne'), ('russe', 'russe'),
    ('japonais', 'japonaise'), ('chinois', 'chinoise'), ('portugais', 'portugaise'),
    ('néerlandais', 'néerlandaise'), ('autrichien', 'autrichienne'), ('polonais', 'polonaise'),
    ('suédois', 'suédoise'), ('grec', 'grecque'), ('argentin', 'argentine'), ('brésilien', 'brésilienne'),
    ('mexicain', 'mexicaine'), ('irlandais', 'irlandaise'), ('écossais', 'écossaise'),
    ('australien', 'australienne'), ('algérien', 'algérienne'), ('marocain', 'marocaine'),
    ('tunisien', 'tunisienne'), ('sénégalais', 'sénégalaise'), ('danois', 'danoise'),
    ('norvégien', 'norvégienne'), ('finlandais', 'finlandaise'), ('hongrois', 'hongroise'),
    ('tchèque', 'tchèque'), ('roumain', 'roumaine'), ('turc', 'turque'), ('iranien', 'iranienne'),
    ('indien', 'indienne'), ('égyptien', 'égyptienne'), ('israélien', 'israélienne'),
    ('québécois', 'québécoise'), ('luxembourgeois', 'luxembourgeoise'), ('ukrainien', 'ukrainienne'),
    ('serbe', 'serbe'), ('croate', 'croate'), ('colombien', 'colombienne'), ('chilien', 'chilienne'),
    ('camerounais', 'camerounaise'), ('libanais', 'libanaise'), ('congolais', 'congolaise'),
]
METIERS = """acteur actrice chanteur chanteuse peintre sculpteur sculptrice écrivain écrivaine poète poétesse
romancier romancière compositeur compositrice musicien musicienne réalisateur réalisatrice footballeur
footballeuse cycliste architecte philosophe mathématicien mathématicienne physicien physicienne chimiste
médecin historien historienne journaliste photographe dessinateur dessinatrice scénariste dramaturge
militaire général officier diplomate juriste avocat avocate prêtre évêque cardinal pape explorateur
exploratrice navigateur aviateur aviatrice ingénieur pianiste violoniste guitariste rappeur rappeuse
humoriste animateur animatrice producteur productrice danseur danseuse athlète nageur nageuse skieur
skieuse boxeur pilote entraîneur sociologue économiste linguiste théologien astronome botaniste
naturaliste biologiste géographe archéologue éditeur éditrice illustrateur illustratrice résistant
résistante mannequin""".split()
METIERS_COMPOSES = ["homme politique", "femme politique", "homme d'État", "femme d'État", "chef d'orchestre",
                    "joueur de tennis", "joueuse de tennis", "joueur de basket-ball", "joueur de rugby",
                    "joueur de hockey sur glace", "joueur de handball", "auteur-compositeur-interprète",
                    "autrice-compositrice-interprète", "homme d'affaires", "femme d'affaires"]
PAR = {'réalisé': "Qui a réalisé {x} ?", 'écrit': "Qui a écrit {x} ?", 'composé': "Qui a composé {x} ?",
       'dessiné': "Qui a dessiné {x} ?", 'fondé': "Qui a fondé {x} ?", 'créé': "Qui a créé {x} ?",
       'développé': "Qui a développé {x} ?", 'construit': "Qui a construit {x} ?", 'conçu': "Qui a conçu {x} ?"}

TITRES_A_EVITER = re.compile(r"^(Liste|Histoire|Chronologie|Géographie|Économie|Démographie|Politique) ")

# « Panthéon » devient « le Panthéon » si le passage commence comme ça
def sujet(titre, texte):
    x = re.sub(r'\s*\(.*\)', '', titre).strip()
    m = re.match(r"(Les |Le |La |L'|L’)" + re.escape(x) + r"\b", texte, re.IGNORECASE)
    if m:
        return m.group(1).lower().replace('’', "'") + texte[m.end(1):m.end()]
    # « La est un fruit » : le dump a perdu le titre, on le remet en minuscule derrière l'article
    m = re.match(r"(Les|Le|La) (?:est|sont)\b", texte)
    if m:
        return m.group(1).lower() + ' ' + x[:1].lower() + x[1:]
    return x

def maj(texte):
    return texte[:1].upper() + texte[1:]

def de(x):
    for article, contracte in (('le ', 'du '), ('les ', 'des '), ('la ', 'de la '), ("l'", "de l'")):
        if x.lower().startswith(article):
            return contracte + x[len(article):]
    return ("d'" if normaliser(x[:1]) in 'aeiouyh' else 'de ') + x

def phrase_de(texte, position):
    point = texte.rfind('. ', 0, position)
    debut = point + 2 if point >= 0 else 0
    fin = texte.find('. ', position)
    phrase = texte[debut:fin + 1 if fin >= 0 else len(texte)].strip()
    return phrase if 30 <= len(phrase) <= 220 else None

# --- Les faits qu'on sait repérer ---
# Le texte de Wikipédia a perdu ses dates et ses nombres : « un peintre français du » est une réponse coupée
COUPEE = re.compile(r"\b(de|du|des|la|le|les|à|au|aux|en|et|par|un|une|d'|l')$")

def faits(titre, texte):
    if TITRES_A_EVITER.match(titre):
        return []
    x = sujet(titre, texte)
    X = maj(x)
    personne = bool(re.search(r"\bnée?\b", texte[:250]))
    trouves = []

    def ajouter(fait, pool, questions, reponse, courte, position):
        reponse = reponse.strip(" '’-")
        if len(reponse) < 2 or len(reponse) > 45 or cle(reponse) in cle(x) or cle(x) in cle(reponse):
            return
        if COUPEE.search(reponse):
            return
        trouves.append({'fait': fait, 'pool': pool, 'questions': [maj(q) for q in questions], 'reponse': reponse,
                        'courte': maj(courte), 'justification': phrase_de(texte, position)})

    # Rien entre « né » et « à » qui contienne un autre « à » : « née à et morte à Neuilly » n'est pas une naissance
    m = re.search(r"\bné(e)?(?: [^,.;()à]{0,30}?)? à " + NOM, texte)
    if m:
        e = m.group(1) or ''
        ajouter('naissance', 'ville', [f"Où est né{e} {x} ?", f"Dans quelle ville est né{e} {x} ?",
                                       f"Quelle est la ville natale {de(x)} ?"],
                m.group(2), f"{x} est né{e} à {m.group(2)}.", m.start())
    m = re.search(r"\b(mort|morte|décédée?)(?: [^,.;()à]{0,30}?)? à " + NOM, texte)
    if m:
        participe = 'morte' if m.group(1) in ('morte', 'décédée') else 'mort'
        ajouter('deces', 'ville', [f"Où est {participe} {x} ?", f"Dans quelle ville est {participe} {x} ?"],
                m.group(2), f"{x} est {participe} à {m.group(2)}.", m.start())
    m = re.search(r"département (de la |de l'|d'|du |des |de )" + NOM, texte)
    if m and not personne:
        ajouter('departement', 'departement', [f"Dans quel département se trouve {x} ?",
                                               f"{x} se trouve dans quel département ?"],
                m.group(2), f"{x} se trouve dans le département {m.group(1)}{m.group(2)}.", m.start())
    m = re.search(r"en région " + NOM, texte)
    if m and not personne:
        ajouter('region', 'region', [f"Dans quelle région se trouve {x} ?", f"{x} est dans quelle région ?"],
                m.group(1), f"{x} se trouve en région {m.group(1)}.", m.start())
    m = re.search(r"\b(fondée?|créée?)s? en (\d{4})\b", texte)
    if m:
        participe = m.group(1)
        pronom = 'elle' if participe.endswith('ée') else 'il'
        ajouter('fondation', 'annee', [f"En quelle année {x} a-t-{pronom} été {participe} ?",
                                       f"De quand date la fondation {de(x)} ?"],
                m.group(2), f"{x} a été {participe} en {m.group(2)}.", m.start())
    m = re.search(r"\b(sortie?)(?: [^,.;()]{0,25}?)? en (\d{4})\b", texte)
    if m:
        ajouter('sortie', 'annee', [f"En quelle année est {m.group(1)} {x} ?", f"{x} est {m.group(1)} en quelle année ?"],
                m.group(2), f"{x} est {m.group(1)} en {m.group(2)}.", m.start())
    m = re.search(r"\b(réalisé|écrit|composé|dessiné|fondé|créé|développé|construit|conçu)e?s? (?:et [a-zé]+ )?par "
                  + NOM, texte)
    if m and not re.match(r"(La|Le|Les|L'|Un|Une)\b", m.group(2)):
        verbe = m.group(1)
        pool = 'entreprise' if verbe == 'développé' else 'personne'
        ajouter(f"auteur_{verbe}", pool, [PAR[verbe].format(x=x)], m.group(2),
                f"{x} a été {verbe} par {m.group(2)}.", m.start())
    m = re.search(r"(?:[Ss]a capitale(?: et (?:sa )?plus grande ville)? est|pour capitale|dont la capitale est) " + NOM, texte)
    if m:
        ajouter('capitale', 'ville', [f"{X} : quelle est sa capitale ?", f"Quelle est la capitale {de(x)} ?"],
                m.group(1), f"La capitale {de(x)} est {m.group(1)}.", m.start())
    m = re.search(r"(?:[Ss]on chef-lieu est|pour chef-lieu|dont le chef-lieu est) " + NOM, texte)
    if m:
        ajouter('chef-lieu', 'ville', [f"Quel est le chef-lieu {de(x)} ?"],
                m.group(1), f"Le chef-lieu {de(x)} est {m.group(1)}.", m.start())
    m = re.search(r"[Ss]es habitants sont (?:appelés|nommés) (?:les )?([A-ZÉÈ][\w\-]+)", texte)
    if m and not personne:
        ajouter('gentile', 'gentile', [f"Comment appelle-t-on les habitants {de(x)} ?",
                                       f"Quel est le nom des habitants {de(x)} ?"],
                m.group(1), f"Les habitants {de(x)} sont appelés les {m.group(1)}.", m.start())
    m = re.search(r"\b(surnommée?|dite?) (?:le |la |l')?« ([^»]{2,40}?) »", texte)
    if m and not re.match(r"(de|d'|du|des) ", m.group(2)):
        ajouter('surnom', 'surnom', [f"Quel est le surnom {de(x)} ?", f"Comment surnomme-t-on {x} ?"],
                m.group(2), f"{x} est surnommé{'e' if m.group(1).endswith('e') else ''} « {m.group(2)} ».", m.start())
    nat = '|'.join(sorted({n for paire in NATIONALITES for n in paire}, key=len, reverse=True))
    metiers = '|'.join(sorted(METIERS_COMPOSES + METIERS, key=len, reverse=True))
    m = re.search(rf"\best (un|une) ({metiers})\b([^,.;()]{{0,60}}?) ({nat})\b", texte)
    if m:
        article, metier, nationalite = m.group(1), m.group(2), m.group(4)
        feminin = article == 'une'
        ajouter('metier', 'metier_' + article, [f"Quel est le métier {de(x)} ?", f"Que fait {x} dans la vie ?"],
                metier, f"{x} est {article} {metier}.", m.start())
        ajouter('nationalite', 'nationalite_' + ('f' if feminin else 'm'),
                [f"Quelle est la nationalité {de(x)} ?", f"{x} est de quelle nationalité ?"],
                nationalite, f"{x} est {nationalite}.", m.start())
    m = re.match(r"^[^.]{0,120}? est (un|une) ([^,.;()«»]{3,70}?)(?=[,.;(]| qui | dont | situé| née?\b| fondé| créé| sorti| de \d|$)",
                 texte)
    if m and len(m.group(2).split()) <= 8:
        definition = f"{m.group(1)} {m.group(2).strip()}"
        ajouter('definition', f"definition_{'personne' if personne else 'chose'}_{m.group(1)}",
                [f"Qu'est-ce que {x} ?", f"C'est quoi {x} ?"] if not re.search(r"\bnée?\b", texte[:200])
                else [f"Qui est {x} ?", f"Qui était {x} ?"],
                definition, f"{x} est {definition}.", m.start())
    return trouves

# --- Distracteurs du même genre ---
def distracteurs(fait, reponse, pools):
    if fait['pool'] == 'annee':
        annee = int(reponse)
        ecarts = random.sample([d for d in range(-40, 41) if d and annee + d <= 2024], 3)
        return [str(annee + d) for d in ecarts]
    pris, cles = [], {cle(reponse)}
    for candidat in random.sample(pools[fait['pool']], min(60, len(pools[fait['pool']]))):
        c = cle(candidat)
        if c and all(c not in d and d not in c for d in cles):
            pris.append(candidat)
            cles.add(c)
        if len(pris) == 3:
            return pris
    return None

def ngrammes(texte, n=N_GRAMME):
    m = re.findall(r'[a-z0-9]+', normaliser(texte))
    return {' '.join(m[i:i + n]) for i in range(len(m) - n + 1)}

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    random.seed(2026)
    with open(chemins.QCM, encoding='utf-8') as f:
        qcm = [json.loads(ligne) for ligne in f]
    titres_qcm = {cle_titre(q['article']) for q in qcm}
    ngrammes_qcm = set().union(*(ngrammes(q['question']) for q in qcm))
    ngrammes_extraits_qcm = set().union(*(ngrammes(q['passage']) for q in qcm))

    recherche = Recherche()
    longueurs = np.load(f"{chemins.RECHERCHE}/articles.npy")
    sources = []
    for i in np.flatnonzero(longueurs >= LONGUEUR_MIN_ARTICLE):
        titre, texte = recherche.passage(int(i))
        if cle_titre(titre) in titres_qcm:
            continue
        sources.append((int(i), titre, texte))
    print(f"{len(sources):,} articles de plus de {LONGUEUR_MIN_ARTICLE:,} caractères, hors QCM v2")

    candidats = []
    pools = collections.defaultdict(set)
    for i, titre, texte in sources:
        for fait in faits(titre, texte):
            candidats.append((i, titre, texte, fait))
            pools[fait['pool']].add(fait['reponse'])
    # Un département ou une région cités dans moins de trois articles, c'est souvent une erreur de lecture
    # (« département de Colombie »), on ne s'en sert pas comme distracteur
    comptes = collections.Counter((c[3]['pool'], c[3]['reponse']) for c in candidats)
    pools = {k: sorted(r for r in v if k not in ('departement', 'region') or comptes[(k, r)] >= 3)
             for k, v in pools.items()}
    print(f"{len(candidats):,} faits repérés : " + ', '.join(
        f"{k} {v}" for k, v in collections.Counter(c[3]['fait'].split('_')[0] for c in candidats).most_common()))
    random.shuffle(candidats)

    types = [t for t, part in PARTS.items() for _ in range(round(NB_EXEMPLES * part))]
    random.shuffle(types)
    positions = [n % 4 for n in range(NB_EXEMPLES)]
    random.shuffle(positions)

    # Un homonyme (« Vox » le parti et « Vox » le média) rend la question ambiguë : on l'écarte quand le titre
    # source a besoin d'une parenthèse pour être compris, et toujours quand le document ne doit pas répondre
    def voisins(question, i, titre_source, reponse, nombre, sans_homonyme):
        trouves = []
        for _, j in recherche.chercher(question, n=10):
            titre, texte = recherche.passage(j)
            homonyme = cle_titre(titre) == cle_titre(titre_source)
            if j == i or cle_titre(titre) in titres_qcm or cle(reponse) in cle(texte) \
                    or ngrammes(texte) & ngrammes_extraits_qcm \
                    or homonyme and (sans_homonyme or '(' in titre_source):
                continue
            trouves.append(texte)
            if len(trouves) == nombre:
                break
        return trouves if len(trouves) == nombre else None

    exemples, par_fait, par_article = [], collections.Counter(), collections.Counter()
    rejets, deja = collections.Counter(), collections.defaultdict(set)
    # Deuxième tour : un fait déjà servi peut resservir une fois, mais dans un autre type d'exemple
    for i, titre, texte, fait in candidats + random.sample(candidats, len(candidats)):
        if len(exemples) == len(types):
            break
        nom = fait['fait'].split('_')[0]
        type_ = types[len(exemples)]
        cle_fait = (i, fait['fait'])
        if par_article[i] >= 3 or len(deja[cle_fait]) >= 2 or type_ in deja[cle_fait] \
                or par_fait[nom] >= MAX_PAR_FAIT.get(nom, 1.0) * NB_EXEMPLES:
            continue
        question = random.choice(fait['questions'])
        if ngrammes(question) & ngrammes_qcm:
            rejets['8-gramme commun avec une question du QCM'] += 1
            continue
        if ngrammes(texte) & ngrammes_extraits_qcm:
            rejets['passage qui recopie un extrait du QCM'] += 1
            continue
        exemple = {'type': type_, 'fait': fait['fait'], 'article': titre}
        qcm_ici = type_ == 'qcm' or type_ == 'plusieurs' and random.random() < 0.5

        if type_ == 'sans_reponse':
            documents = voisins(question, i, titre, fait['reponse'], 2 if random.random() < PART_DEUX_DOCUMENTS else 1, True)
            reponse = random.choice(AUCUNE_REPONSE)
        elif type_ == 'plusieurs':
            autres = voisins(question, i, titre, fait['reponse'], random.choice((1, 2)), False)
            documents = None if autres is None else random.sample(autres + [texte], len(autres) + 1)
        elif type_ == 'qcm' and random.random() < PART_QCM_SANS_DOCUMENT:
            documents, exemple['type'] = [], 'qcm_sans_document'
        else:
            autres = voisins(question, i, titre, fait['reponse'], 1, False) if random.random() < PART_DEUX_DOCUMENTS else []
            # Comme dans chat.py : le meilleur document en dernier
            documents = None if autres is None else autres + [texte]
        if documents is None:
            rejets['pas assez de documents voisins sans la réponse'] += 1
            continue

        if qcm_ici:
            faux = distracteurs(fait, fait['reponse'], pools)
            if faux is None:
                rejets['pas assez de distracteurs'] += 1
                continue
            p = positions[sum('choix' in e for e in exemples)]
            choix = faux[:p] + [fait['reponse']] + faux[p:]
            exemple.update(choix=choix, reponse_juste=p)
            message = avec_les_choix({'question': question, 'choix': choix})
            # Le texte du choix d'abord, recopié du document. Avec « B) Lyon. » le modèle devait choisir la lettre
            # avant d'avoir lu, il la tirait au hasard (22 % de lettres justes) et ne recopiait plus le document
            reponse = f"{maj(fait['reponse'].rstrip('.'))}."
            if documents and fait['justification'] and random.random() < PART_JUSTIFICATION:
                reponse += ' ' + fait['justification']
        else:
            message = question
            if type_ != 'sans_reponse':
                reponse = fait['courte']
        exemple.update(question=message, reponse=reponse, documents=documents)
        exemples.append(exemple)
        par_fait[nom] += 1
        par_article[i] += 1
        deja[cle_fait].add(type_)

    # --- Contrôle : rien du QCM v2 ne doit se retrouver dans les exemples ---
    titres_utilises = {cle_titre(e['article']) for e in exemples}
    assert not titres_utilises & titres_qcm, "un article du QCM v2 sert de source"
    assert not any(ngrammes(e['question']) & ngrammes_qcm for e in exemples), "8-gramme commun avec le QCM v2"
    assert not any(ngrammes(d) & ngrammes_extraits_qcm for e in exemples for d in e['documents']), \
        "un document recopie un extrait du QCM v2"

    with open(chemins.LECTURE, 'w', encoding='utf-8') as f:
        for e in exemples:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')
    print(f"\n{len(exemples):,} exemples -> {chemins.LECTURE}")
    print("types      :", dict(collections.Counter(e['type'] for e in exemples)))
    print("faits      :", dict(par_fait.most_common()))
    print("positions  :", dict(sorted(collections.Counter(e['reponse_juste'] for e in exemples if 'choix' in e).items())))
    print("documents  :", dict(sorted(collections.Counter(len(e['documents']) for e in exemples).items())))
    print("rejetés    :", dict(rejets))
    print(f"contrôle QCM v2 : 0 article commun, 0 question avec un {N_GRAMME}-gramme commun, "
          f"0 document qui recopie un extrait ({len(titres_utilises):,} articles sources)")

if __name__ == '__main__':
    main()
