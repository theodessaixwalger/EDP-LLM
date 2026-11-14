import argparse
import collections
import datetime
import json
import math
import os
import random
import re
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from choix import vraisemblance as choix_vraisemblance
from model import MonLLM, bloquer_repetitions
from recherche import SEUIL, Recherche, ligne_document, mots, normaliser
from tokeniseur import decoupage
import chemins

# python evaluer_qcm.py --checkpoint modeles/checkpoint_chat.pt modeles/checkpoint_dpo.pt
# python evaluer_qcm.py --comparer resultats/a.json resultats/b.json  (tests appariés entre deux runs)
QCM = chemins.QCM
RESULTATS = 'resultats'
UTILISATEUR = "Utilisateur : "
ASSISTANT = "Assistant : "
# Mêmes réglages que chat.py
TEMPERATURE = 0.5
LONGUEUR_MAX_REPONSE = 400
K_RAPPEL = (1, 2, 5, 10)
NB_BOOTSTRAP = 2000
HASARD = 0.25

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def arguments():
    parseur = argparse.ArgumentParser()
    parseur.add_argument('--checkpoint', nargs='+',
                         default=[f for f in (chemins.MODELE_CHAT, chemins.MODELE_DPO) if os.path.exists(f)])
    parseur.add_argument('--n-tirages', type=int, default=5)
    parseur.add_argument('--limite', type=int, default=None)
    parseur.add_argument('--sans-documents', action='store_true',
                         help="n'évalue que la condition sans documents (pas de recherche)")
    parseur.add_argument('--k', type=int, default=2, help="nombre de documents injectés (2 dans chat.py)")
    parseur.add_argument('--seuil', type=float, default=SEUIL)
    parseur.add_argument('--sans-seuil', action='store_true', help="toujours injecter k documents")
    parseur.add_argument('--sans-generation', action='store_true', help="vraisemblance seulement")
    parseur.add_argument('--graine', type=int, default=0)
    parseur.add_argument('--qcm', default=QCM)
    parseur.add_argument('--etiquette', default='', help="ajoutée au nom du fichier de résultats")
    parseur.add_argument('--comparer', nargs=2, metavar=('REFERENCE', 'AUTRE'),
                         help="compare deux fichiers de résultats question par question, sans rien relancer")
    args = parseur.parse_args()
    if args.sans_seuil:
        args.seuil = 0.0
    return args

# --- Reconnaître un choix dans un texte ---
DETERMINANT = re.compile(r"^(le|la|les|l|un|une|des|du|de|d|au|aux|a|en|par|dans|sur|celui de|celle de|ceux de) ")

def cle(texte):
    texte = ' '.join(re.findall(r'[a-z0-9]+', normaliser(texte)))
    while DETERMINANT.match(texte):
        texte = DETERMINANT.sub('', texte, count=1)
    return texte

# Un choix est cité si on retrouve son texte entier, ou un de ses mots que les autres choix n'ont pas
def mots_propres(choix):
    tous = [set(mots(c)) for c in choix]
    return [tous[i] - set().union(*(tous[j] for j in range(len(choix)) if j != i)) for i in range(len(choix))]

def cites(texte, choix):
    plat, ensemble = f" {cle(texte)} ", set(mots(texte))
    return [i for i, (c, propres) in enumerate(zip(choix, mots_propres(choix)))
            if f" {cle(c)} " in plat or propres & ensemble]

# Réponse au format « B) Lyon. Justification... » ou « Lyon. Justification... » : la justification peut citer
# un autre choix, donc on lit seulement ce qui suit la lettre, ou la première phrase si c'est exactement un
# des choix. Sinon on lit toute la réponse (une première phrase qui cite juste un choix au passage ne suffit pas)
FORMAT_LETTRE = re.compile(r"^\s*([ABCD])\)\s*(.*?)(?:\.(?:\s|$)|$)", re.S)
PREMIERE_PHRASE = re.compile(r"^\s*(.*?)(?:\.(?:\s|$)|$)", re.S)

def extraire(reponse, choix):
    lettre = FORMAT_LETTRE.match(reponse)
    if lettre:
        trouves = cites(lettre.group(2), choix)
        if not trouves:
            return 'ABCD'.index(lettre.group(1))
        return trouves[0] if len(trouves) == 1 else None
    premiere = cle(PREMIERE_PHRASE.match(reponse).group(1))
    exacts = [i for i, c in enumerate(choix) if cle(c) == premiere]
    if len(exacts) == 1:
        return exacts[0]
    trouves = cites(reponse, choix)
    return trouves[0] if len(trouves) == 1 else None

def figure_dans(texte, choix, i):
    propres = mots_propres(choix)[i]
    return f" {cle(choix[i])} " in f" {cle(texte)} " or bool(propres) and propres <= set(mots(texte))

# --- Statistiques ---
def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    marge = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - marge), min(1.0, centre + marge))

def bootstrap(valeurs, graine):
    valeurs = np.asarray(valeurs, dtype=float)
    if len(valeurs) == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(graine)
    moyennes = valeurs[rng.integers(0, len(valeurs), (NB_BOOTSTRAP, len(valeurs)))].mean(axis=1)
    return tuple(float(x) for x in np.percentile(moyennes, [2.5, 97.5]))

def mcnemar(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    seul_a, seul_b = int((a & ~b).sum()), int((~a & b).sum())
    n = seul_a + seul_b
    # Test exact : sous l'hypothèse nulle chaque désaccord penche d'un côté avec une chance sur deux
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(seul_a, seul_b) + 1)) / 2 ** n) if n else 1.0
    return {'seul_a': seul_a, 'seul_b': seul_b, 'p': p}

def bootstrap_apparie(a, b, graine):
    d = np.asarray(b, float) - np.asarray(a, float)
    rng = np.random.default_rng(graine)
    moyennes = d[rng.integers(0, len(d), (NB_BOOTSTRAP, len(d)))].mean(axis=1)
    p = min(1.0, 2 * min((moyennes <= 0).mean(), (moyennes >= 0).mean()))
    bas, haut = np.percentile(moyennes, [2.5, 97.5])
    return {'difference': float(d.mean()), 'ic95': [float(bas), float(haut)], 'p': float(p)}

def score_binaire(valeurs):
    k, n = int(sum(valeurs)), len(valeurs)
    return {'score': k / n if n else 0.0, 'ic95': list(wilson(k, n)), 'n': n}

def score_moyen(valeurs, graine):
    return {'score': float(np.mean(valeurs)) if len(valeurs) else 0.0,
            'ic95': list(bootstrap(valeurs, graine)), 'n': len(valeurs)}

# --- Recherche (ne dépend pas du modèle, faite une seule fois) ---
def rechercher(questions, k, seuil):
    recherche = Recherche()
    trouves = []
    for q in questions:
        documents = recherche.documents_pour(q['question'], n=k, seuil=seuil)
        classement = [recherche.passage(i)[0] for _, i in recherche.chercher(q['question'], n=max(K_RAPPEL))]
        texte = ' '.join(p for _, p in documents)
        trouves.append({
            'documents': documents,
            'titres': [t for t, _ in reversed(documents)],
            'article_trouve': q['article'] in [t for t, _ in documents],
            'rappel': {k: q['article'] in classement[:k] for k in K_RAPPEL},
            'reponse_dans_documents': figure_dans(texte, q['choix'], q['reponse']),
        })
    return trouves

# --- Modèle ---
class Evaluateur:
    def __init__(self, fichier):
        self.modele = MonLLM.charger(fichier, device)
        self.fenetre = self.modele.taille_contexte
        self.tok = decoupage(self.modele.embed_tokens.num_embeddings)

    def propre(self, texte):
        return self.tok.nettoyer(texte)[0]

    def contexte(self, question, documents):
        lignes = ''.join(ligne_document(self.propre(p)) for _, p in documents)
        return lignes + UTILISATEUR + self.propre(question) + '\n' + ASSISTANT

    @torch.no_grad()
    def vraisemblance(self, contexte, choix):
        # Le même calcul que le chatbot quand on lui pose une question à choix
        return choix_vraisemblance(self.modele, self.tok, contexte, [self.propre(c) for c in choix])

    @torch.no_grad()
    def generer(self, contexte, n):
        idx = torch.tensor([self.tok.encoder(contexte)] * n, dtype=torch.long, device=device)
        reponses, finies = [[] for _ in range(n)], [False] * n
        for _ in range(LONGUEUR_MAX_REPONSE):
            derniers = self.modele(idx[:, -self.fenetre:])[0][:, -1]
            lignes = torch.stack([bloquer_repetitions(derniers[i], reponses[i], self.tok.anti_boucle)
                                  for i in range(n)])
            suivants = torch.multinomial(F.softmax(lignes / TEMPERATURE, dim=-1), 1).view(-1).tolist()
            for i, s in enumerate(suivants):
                if finies[i]:
                    suivants[i] = self.tok.saut_de_ligne
                elif s == self.tok.saut_de_ligne:
                    finies[i] = True
                else:
                    reponses[i].append(s)
            if all(finies):
                break
            idx = torch.cat((idx, torch.tensor(suivants, device=device).view(n, 1)), dim=1)
        return [self.tok.decoder(r) for r in reponses]

# En génération on écrit les quatre choix dans la question, sinon le hasard ne ferait pas 25 %
def avec_les_choix(q):
    return f"{q['question']} ({', '.join(q['choix'][:-1])} ou {q['choix'][-1]})"

def vote(extraits):
    comptes = collections.Counter(e for e in extraits if e is not None).most_common()
    if not comptes or len(comptes) > 1 and comptes[0][1] == comptes[1][1]:
        return None
    return comptes[0][0]

def evaluer(fichier, questions, conditions, trouves, args):
    ev = Evaluateur(fichier)
    details = []
    debut = time.time()
    for n, q in enumerate(questions):
        ligne = {'id': q['id']}
        for condition in conditions:
            documents = trouves[n]['documents'] if condition == 'avec_documents' else []
            sommes, normalisees = ev.vraisemblance(ev.contexte(q['question'], documents), q['choix'])
            ligne[condition] = {
                'somme': sommes, 'normalisee': normalisees,
                'choix_somme': int(np.argmax(sommes)), 'choix_normalisee': int(np.argmax(normalisees)),
                'tokens_documents': len(ev.tok.encoder(ev.contexte('', documents))) - len(ev.tok.encoder(ev.contexte('', []))),
            }
            if args.sans_generation:
                continue
            torch.manual_seed(args.graine * 1_000_003 + int(q['id'].split('_')[1]))
            reponses = ev.generer(ev.contexte(avec_les_choix(q), documents), args.n_tirages)
            extraits = [extraire(r, q['choix']) for r in reponses]
            ligne[condition].update({'reponses': reponses, 'extraits': extraits, 'vote': vote(extraits)})
        details.append(ligne)
        if (n + 1) % 25 == 0:
            ecoule = time.time() - debut
            print(f"    {n + 1}/{len(questions)} questions ({ecoule:.0f}s, "
                  f"reste ~{ecoule / (n + 1) * (len(questions) - n - 1):.0f}s)", flush=True)
    del ev
    torch.cuda.empty_cache()
    return details

# --- Mesures tirées des détails ---
METRIQUES = ('vraisemblance_normalisee', 'vraisemblance_somme', 'generation_moyenne', 'generation_vote')

def par_question(q, d):
    juste = q['reponse']
    valeurs = {
        'vraisemblance_normalisee': d['choix_normalisee'] == juste,
        'vraisemblance_somme': d['choix_somme'] == juste,
    }
    if 'extraits' in d:
        valeurs['generation_moyenne'] = float(np.mean([e == juste for e in d['extraits']]))
        valeurs['generation_vote'] = d['vote'] == juste
    return valeurs

def scores(valeurs, graine):
    return {m: (score_moyen([v[m] for v in valeurs], graine) if m == 'generation_moyenne'
                else score_binaire([v[m] for v in valeurs])) for m in METRIQUES if valeurs and m in valeurs[0]}

def resumer(questions, details, conditions, trouves, graine):
    resume = {}
    for condition in conditions:
        valeurs = [par_question(q, d[condition]) for q, d in zip(questions, details)]
        bloc = {'global': scores(valeurs, graine),
                'tokens_documents': float(np.mean([d[condition]['tokens_documents'] for d in details])),
                'par_categorie': {}, 'par_type': {}}
        if 'extraits' in details[0][condition]:
            extraits = [e for d in details for e in d[condition]['extraits']]
            bloc['sans_reponse_reconnue'] = sum(e is None for e in extraits) / len(extraits)
            bloc['position_citee'] = {lettre: sum(e == i for e in extraits) / len(extraits)
                                      for i, lettre in enumerate('ABCD')}
        for champ, cible in (('categorie', 'par_categorie'), ('type', 'par_type')):
            for valeur in sorted({q[champ] for q in questions}):
                sous = [v for q, v in zip(questions, valeurs) if q[champ] == valeur]
                bloc[cible][valeur] = scores(sous, graine)
        if condition == 'avec_documents':
            bloc['recherche'] = {}
            for nom, cle_ in (('article_trouve', 'article_trouve'), ('reponse_dans_documents', 'reponse_dans_documents')):
                for present in (True, False):
                    sous = [v for v, t in zip(valeurs, trouves) if t[cle_] == present]
                    sans = [par_question(q, d['sans_documents']) for q, d, t in zip(questions, details, trouves)
                            if t[cle_] == present] if 'sans_documents' in conditions else []
                    bloc['recherche'][f"{nom}={'oui' if present else 'non'}"] = {
                        'avec_documents': scores(sous, graine), 'sans_documents': scores(sans, graine) if sans else None}
        resume[condition] = bloc
    if len(conditions) == 2:
        resume['documents_vs_sans'] = comparer(questions, details, details, 'sans_documents', 'avec_documents', graine)
    return resume

def comparer(questions, details_a, details_b, condition_a, condition_b, graine):
    a = [par_question(q, d[condition_a]) for q, d in zip(questions, details_a)]
    b = [par_question(q, d[condition_b]) for q, d in zip(questions, details_b)]
    sortie = {}
    for m in METRIQUES:
        if m not in a[0] or m not in b[0]:
            continue
        va, vb = [x[m] for x in a], [x[m] for x in b]
        sortie[m] = bootstrap_apparie(va, vb, graine)
        if m != 'generation_moyenne':
            sortie[m]['mcnemar'] = mcnemar(va, vb)
    return sortie

# --- Affichage ---
NOMS = {'vraisemblance_normalisee': 'vraisemblance / longueur', 'vraisemblance_somme': 'vraisemblance (somme)',
        'generation_moyenne': 'génération, moyenne', 'generation_vote': 'génération, vote'}

def pourcent(s):
    return f"{s['score']:6.1%} [{s['ic95'][0]:5.1%}-{s['ic95'][1]:5.1%}]"

def afficher(rapport):
    noms = list(rapport['checkpoints'])
    courts = [os.path.basename(n) for n in noms]
    conditions = rapport['conditions']
    n = rapport['nb_questions']
    print(f"\n{'=' * 100}\nQCM v2 : {n} questions, 4 choix (hasard = 25 %), IC à 95 % "
          f"(Wilson, bootstrap pour la moyenne des {rapport['n_tirages']} tirages)\n")
    largeur = 31
    for condition in conditions:
        blocs = [rapport['checkpoints'][c][condition] for c in noms]
        print(f"--- {condition.replace('_', ' ')} ---")
        print(f"  {'':26}" + ''.join(f"{c:>{largeur}}" for c in courts))
        for m in METRIQUES:
            if m in blocs[0]['global']:
                print(f"  {NOMS[m]:26}" + ''.join(f"{pourcent(b['global'][m]):>{largeur}}" for b in blocs))
        if 'sans_reponse_reconnue' in blocs[0]:
            print(f"  {'un seul choix cité (gén.)':26}" + ''.join(
                f"{1 - b['sans_reponse_reconnue']:>{largeur}.1%}" for b in blocs))
            print(f"  {'choix cités A/B/C/D (gén.)':26}" + ''.join(
                f"{' '.join(f'{v:.0%}' for v in b['position_citee'].values()):>{largeur}}" for b in blocs))
        if condition == 'avec_documents':
            print(f"  {'tokens de documents':26}" + ''.join(f"{b['tokens_documents']:>{largeur}.0f}" for b in blocs))
        print()
    principal = 'vraisemblance_normalisee'
    for champ in ('par_categorie', 'par_type'):
        print(f"--- {champ.replace('_', ' ')} (vraisemblance / longueur) ---")
        groupes = rapport['checkpoints'][noms[0]][conditions[0]][champ]
        entete = ''.join(f"{f'{c[:14]} {cond[:4]}':>22}" for c in courts for cond in conditions)
        print(f"  {'':20}{'n':>4}{entete}")
        for g in groupes:
            nb = groupes[g][principal]['n']
            print(f"  {g:20}{nb:>4}" + ''.join(
                f"{rapport['checkpoints'][c][cond][champ][g][principal]['score']:>22.1%}"
                for c in noms for cond in conditions))
        print()
    if 'avec_documents' in conditions:
        r = rapport['recherche']
        print(f"--- recherche (k = {rapport['k']}, seuil = {rapport['seuil']}) ---")
        print(f"  rappel@k de l'article source : " + ', '.join(f"@{k} {v:.0%}" for k, v in r['rappel'].items()))
        print(f"  article source injecté        : {r['article_injecte']:.0%}")
        print(f"  bonne réponse dans les docs   : {r['reponse_dans_documents']:.0%}")
        print(f"  précision selon ce qui a été trouvé (vraisemblance / longueur, avec docs | sans docs) :")
        for c, court in zip(noms, courts):
            for cas, v in rapport['checkpoints'][c]['avec_documents']['recherche'].items():
                sans = f" | {v['sans_documents'][principal]['score']:6.1%}" if v['sans_documents'] else ''
                print(f"    {court:24} {cas:28} n={v['avec_documents'][principal]['n']:<4}"
                      f"{v['avec_documents'][principal]['score']:6.1%}{sans}")
        print()
    if len(conditions) == 2:
        print("--- avec documents - sans documents (mêmes questions, test apparié) ---")
        for c, court in zip(noms, courts):
            d = rapport['checkpoints'][c]['documents_vs_sans']
            for m, x in d.items():
                test = f"McNemar p={x['mcnemar']['p']:.3f}" if 'mcnemar' in x else f"bootstrap p={x['p']:.3f}"
                print(f"  {court:24} {NOMS[m]:26} {x['difference']:+6.1%} [{x['ic95'][0]:+.1%} ; {x['ic95'][1]:+.1%}]  {test}")
        print()
    for paire, comparaison in rapport.get('comparaisons', {}).items():
        print(f"--- {paire} (mêmes questions, test apparié) ---")
        for condition, d in comparaison.items():
            for m, x in d.items():
                test = (f"McNemar p={x['mcnemar']['p']:.3f} ({x['mcnemar']['seul_a']} vs {x['mcnemar']['seul_b']} désaccords)"
                        if 'mcnemar' in x else f"bootstrap p={x['p']:.3f}")
                print(f"  {condition:15} {NOMS[m]:26} {x['difference']:+6.1%} [{x['ic95'][0]:+.1%} ; {x['ic95'][1]:+.1%}]  {test}")
        print()

# Deux runs déjà faits (autre k, autre modèle...) : on apparie les questions par id et les modèles dans l'ordre
def comparer_fichiers(chemin_a, chemin_b, graine):
    a, b = (json.load(open(c, encoding='utf-8')) for c in (chemin_a, chemin_b))
    with open(a['qcm'], encoding='utf-8') as f:
        questions = {q['id']: q for q in map(json.loads, f)}
    # On relit les réponses générées avec la règle actuelle de extraire(), pour que les deux fichiers
    # soient notés pareil même s'ils ont été faits avec une version plus ancienne
    for r in (a, b):
        for details in r['details'].values():
            for d in details:
                for condition in r['conditions']:
                    if 'reponses' in d[condition]:
                        d[condition]['extraits'] = [extraire(x, questions[d['id']]['choix']) for x in d[condition]['reponses']]
                        d[condition]['vote'] = vote(d[condition]['extraits'])
    modeles_a, modeles_b = list(a['details']), list(b['details'])
    if len(modeles_a) == 1:
        modeles_a = modeles_a * len(modeles_b)
    if len(modeles_b) == 1:
        modeles_b = modeles_b * len(modeles_a)
    for nom_a, nom_b in zip(modeles_a, modeles_b):
        da = {d['id']: d for d in a['details'][nom_a]}
        db = {d['id']: d for d in b['details'][nom_b]}
        ids = [i for i in da if i in db]
        qs = [questions[i] for i in ids]
        print(f"\n--- {nom_b} ({os.path.basename(chemin_b)}) - {nom_a} ({os.path.basename(chemin_a)}), {len(ids)} questions ---")
        for condition in [c for c in a['conditions'] if c in b['conditions']]:
            d = comparer(qs, [da[i] for i in ids], [db[i] for i in ids], condition, condition, graine)
            for m, x in d.items():
                sa = np.mean([par_question(q, da[i][condition])[m] for q, i in zip(qs, ids)])
                test = (f"McNemar p={x['mcnemar']['p']:.3f} ({x['mcnemar']['seul_a']} vs {x['mcnemar']['seul_b']} désaccords)"
                        if 'mcnemar' in x else f"bootstrap p={x['p']:.3f}")
                print(f"  {condition:15} {NOMS[m]:26} {sa:6.1%} -> {sa + x['difference']:6.1%}  "
                      f"{x['difference']:+6.1%} [{x['ic95'][0]:+.1%} ; {x['ic95'][1]:+.1%}]  {test}")

def main():
    args = arguments()
    sys.stdout.reconfigure(encoding='utf-8')
    if args.comparer:
        comparer_fichiers(*args.comparer, args.graine)
        return
    random.seed(args.graine)
    np.random.seed(args.graine)
    with open(args.qcm, encoding='utf-8') as f:
        questions = [json.loads(ligne) for ligne in f][:args.limite]
    conditions = ['sans_documents'] if args.sans_documents else ['sans_documents', 'avec_documents']
    print(f"{len(questions)} questions, {len(args.checkpoint)} modèle(s), conditions : {', '.join(conditions)}, "
          f"{args.n_tirages} tirages, graine {args.graine}")

    trouves = [None] * len(questions)
    if 'avec_documents' in conditions:
        print(f"Recherche des documents (k = {args.k}, seuil = {args.seuil})...")
        trouves = rechercher(questions, args.k, args.seuil)

    rapport = {'date': datetime.datetime.now().isoformat(timespec='seconds'), 'qcm': args.qcm,
               'nb_questions': len(questions), 'n_tirages': 0 if args.sans_generation else args.n_tirages,
               'graine': args.graine, 'temperature': TEMPERATURE, 'k': args.k, 'seuil': args.seuil,
               'conditions': conditions, 'checkpoints': {}, 'details': {}}
    tous_details = {}
    for fichier in args.checkpoint:
        print(f"\n{fichier}")
        details = evaluer(fichier, questions, conditions, trouves, args)
        tous_details[fichier] = details
        rapport['checkpoints'][fichier] = resumer(questions, details, conditions, trouves, args.graine)
        rapport['details'][fichier] = details

    if 'avec_documents' in conditions:
        rapport['recherche'] = {
            'rappel': {k: float(np.mean([t['rappel'][k] for t in trouves])) for k in K_RAPPEL},
            'article_injecte': float(np.mean([t['article_trouve'] for t in trouves])),
            'reponse_dans_documents': float(np.mean([t['reponse_dans_documents'] for t in trouves])),
            'par_question': [{'id': q['id'], 'titres': t['titres'], 'article_trouve': t['article_trouve'],
                              'rappel': t['rappel'], 'reponse_dans_documents': t['reponse_dans_documents']}
                             for q, t in zip(questions, trouves)],
        }
    reference = args.checkpoint[0]
    rapport['comparaisons'] = {
        f"{os.path.basename(autre)} - {os.path.basename(reference)}": {
            condition: comparer(questions, tous_details[reference], tous_details[autre], condition, condition, args.graine)
            for condition in conditions}
        for autre in args.checkpoint[1:]}

    afficher(rapport)
    os.makedirs(RESULTATS, exist_ok=True)
    etiquette = f"_{args.etiquette}" if args.etiquette else ''
    sortie = os.path.join(RESULTATS, f"qcm_{datetime.datetime.now():%Y%m%d_%H%M%S}{etiquette}.json")
    with open(sortie, 'w', encoding='utf-8') as f:
        json.dump(rapport, f, ensure_ascii=False, indent=1)
    print(f"Résultats détaillés -> {sortie}")

if __name__ == '__main__':
    main()
