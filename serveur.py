import argparse
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
import torch.nn.functional as F
from choix import repondre as repondre_par_choix, trouver_choix
from model import MonLLM, bloquer_repetitions
from outils import utiliser as utiliser_outil
from recherche import ligne_document
from recherche_marketing import MODE_PAR_DEFAUT, MODES, RechercheRoutee, etiquette
from tokeniseur import decoupage
import chemins

# Le modèle tourne sur le GPU de ce PC, la page web lui parle via ce serveur (que en local)
HOTE, PORT = '127.0.0.1', 8000

fichier_memoire = chemins.MEMOIRE
fichier_preferences = chemins.PREFERENCES
longueur_max_reponse = 400
temperature = 0.5
temperature_seconde_chance = 0.9

UTILISATEUR = "Utilisateur : "
ASSISTANT = "Assistant : "

device = 'cuda' if torch.cuda.is_available() else 'cpu'

fichier_modele = chemins.modele_le_plus_recent()
modele = MonLLM.charger(fichier_modele, device)
taille_contexte = modele.taille_contexte
tok = decoupage(modele.embed_tokens.num_embeddings)

def longueur(texte):
    return len(tok.encoder(texte))

# Où chercher avant une question : general (Wikipédia), marketing (base marketing) ou auto (les deux)
parser = argparse.ArgumentParser()
parser.add_argument('--recherche', choices=MODES, default=MODE_PAR_DEFAUT)
args = parser.parse_args()
recherche = RechercheRoutee(args.recherche) if os.path.exists(os.path.join(chemins.RECHERCHE, 'positions.npy')) else None

# Un seul modèle sur un seul GPU, les réponses passent une par une
verrou = threading.Lock()

dernier = {'contexte': None, 'reponse': None}

# --- Mémoire et préférences, mêmes fichiers que chat.py ---
def charger_memoire():
    if os.path.exists(fichier_memoire):
        with open(fichier_memoire, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def sauvegarder_memoire(memoire):
    with open(fichier_memoire, 'w', encoding='utf-8') as f:
        json.dump(memoire, f, ensure_ascii=False, indent=2)

def compter_preferences():
    if not os.path.exists(fichier_preferences):
        return 0
    with open(fichier_preferences, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

def enregistrer_preference(contexte, choisie, rejetee):
    if not choisie.strip() or choisie.strip() == rejetee.strip():
        return False
    with open(fichier_preferences, 'a', encoding='utf-8') as f:
        f.write(json.dumps({'contexte': contexte, 'choisie': choisie, 'rejetee': rejetee,
                            'origine': 'web'}, ensure_ascii=False) + '\n')
    return True

memoire = charger_memoire()

# --- Fenêtre de contexte ---
def bloc(echange):
    return UTILISATEUR + echange['utilisateur'] + '\n' + ASSISTANT + echange['assistant'] + '\n'

# Les derniers échanges qui rentrent dans la fenêtre, les plus vieux restent dans le fichier mais il les voit plus
def historique_visible(place):
    texte, nb, occupe = '', 0, 0
    for echange in reversed(memoire):
        taille = longueur(bloc(echange))
        if occupe + taille > place:
            break
        texte, nb, occupe = bloc(echange) + texte, nb + 1, occupe + taille
    return texte, nb

# Les documents servent que pour la question en cours, on les garde pas dans l'historique
def etat_fenetre(question='', documents=''):
    place = taille_contexte - longueur_max_reponse - longueur(documents + question)
    visible, nb = historique_visible(place)
    return {
        'capacite': taille_contexte,
        'reserve': longueur_max_reponse,
        'visible': visible + documents + question,
        'longueur': longueur(visible + documents + question),
        'nb_visibles': nb,
        'nb_total': len(memoire),
        'hors_fenetre': memoire[:len(memoire) - nb],
    }

def nettoyer(message):
    return tok.nettoyer(' '.join(message.split()))

def chercher_documents(message):
    outil = utiliser_outil(message)
    if outil:
        nom, resultat = outil
        return ligne_document(tok.nettoyer(resultat)[0]), [nom]
    if recherche is None:
        return '', []
    trouves = recherche.trouver(message)
    texte = ''.join(ligne_document(tok.nettoyer(document['texte'])[0]) for document in trouves)
    return texte, [etiquette(document) for document in reversed(trouves)]

@torch.no_grad()
def flux_reponse(contexte, chaleur):
    idx = torch.tensor([tok.encoder(contexte)], dtype=torch.long, device=device)
    ids_reponse = []
    for _ in range(longueur_max_reponse):
        logits, _ = modele(idx[:, -taille_contexte:])
        logits = bloquer_repetitions(logits[0, -1], ids_reponse, tok.anti_boucle)
        suivant = torch.multinomial(F.softmax(logits / chaleur, dim=-1), num_samples=1)
        if suivant.item() == tok.saut_de_ligne:
            return
        ids_reponse.append(suivant.item())
        idx = torch.cat((idx, suivant.view(1, 1)), dim=1)
        yield tok.morceaux[suivant.item()]

# Le calcul se fait au premier next(), donc sous le verrou de diffuser()
def flux_choix(contexte, choix, documents):
    yield repondre_par_choix(modele, tok, contexte, choix, documents)

# --- Serveur ---
class Interface(BaseHTTPRequestHandler):
    def log_message(self, *arguments):
        pass

    def repondre_json(self, donnees, code=200):
        corps = json.dumps(donnees, ensure_ascii=False).encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(corps)))
        self.end_headers()
        self.wfile.write(corps)

    def lire_json(self):
        taille = int(self.headers.get('Content-Length', 0))
        return json.loads(self.rfile.read(taille) or b'{}')

    def ligne(self, donnees):
        self.wfile.write((json.dumps(donnees, ensure_ascii=False) + '\n').encode('utf-8'))
        self.wfile.flush()

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'interface.html'), 'rb') as f:
                corps = f.read()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(corps)))
            self.end_headers()
            self.wfile.write(corps)
        elif self.path == '/api/etat':
            self.repondre_json({
                'modele': fichier_modele,
                'device': device,
                'parametres': round(sum(p.numel() for p in modele.parameters()) / 1e6),
                'passages': recherche.nb_passages if recherche else 0,
                'recherche': recherche.mode if recherche else None,
                'vocabulaire': tok.taille_vocab,
                'decoupage': 'caractères' if len(tok.morceaux) == len(tok.alphabet) else 'sous-mots',
                'preferences': compter_preferences(),
                'memoire': memoire,
                'fenetre': etat_fenetre(),
            })
        else:
            self.repondre_json({'erreur': 'inconnu'}, 404)

    def do_POST(self):
        if self.path == '/api/message':
            self.envoyer_message()
        elif self.path == '/api/regenerer':
            self.regenerer()
        elif self.path == '/api/preference':
            self.corriger()
        elif self.path == '/api/oublier':
            memoire.clear()
            sauvegarder_memoire(memoire)
            print("mémoire effacée")
            self.repondre_json({'memoire': memoire, 'fenetre': etat_fenetre()})
        else:
            self.repondre_json({'erreur': 'inconnu'}, 404)

    def diffuser(self, flux):
        self.send_response(200)
        self.send_header('Content-Type', 'application/x-ndjson; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        morceaux = []
        with verrou:
            for morceau in flux:
                morceaux.append(morceau)
                self.ligne({'caractere': morceau})
        return ''.join(morceaux)

    def envoyer_message(self):
        message, inconnus = nettoyer(self.lire_json().get('message', '').strip())
        if not message:
            self.repondre_json({'erreur': 'message vide', 'inconnus': inconnus}, 400)
            return

        question = UTILISATEUR + message + '\n' + ASSISTANT
        # Question à choix : on cherche avec la question seule et on prend le choix le plus probable
        qcm = trouver_choix(message)
        documents, sources = chercher_documents(qcm[0] if qcm else message)
        fenetre = etat_fenetre(question, documents)
        contexte = fenetre['visible']
        if qcm:
            sans_la_liste = contexte[:len(contexte) - len(question)] + UTILISATEUR + qcm[0] + '\n' + ASSISTANT
            flux = flux_choix(sans_la_liste, qcm[1], documents)
        else:
            flux = flux_reponse(contexte, temperature)

        try:
            reponse = self.diffuser(flux)
        except (BrokenPipeError, ConnectionResetError):
            return

        dernier['contexte'], dernier['reponse'] = contexte, reponse
        echange = {'utilisateur': message, 'assistant': reponse}
        if sources:
            echange['sources'] = sources
        memoire.append(echange)
        sauvegarder_memoire(memoire)
        print(f"{message[:50]!r} -> {reponse[:50]!r}" + (f"  (lu : {', '.join(sources)})" if sources else ''))
        self.ligne({'fin': True, 'reponse': reponse, 'inconnus': inconnus, 'sources': sources,
                    'fenetre': etat_fenetre(), 'preferences': compter_preferences()})

    def regenerer(self):
        if dernier['contexte'] is None:
            self.repondre_json({'erreur': 'rien à régénérer'}, 400)
            return
        try:
            proposition = self.diffuser(flux_reponse(dernier['contexte'], temperature_seconde_chance))
        except (BrokenPipeError, ConnectionResetError):
            return
        self.ligne({'fin': True, 'reponse': proposition})

    def corriger(self):
        choisie, inconnus = nettoyer(self.lire_json().get('choisie', '').strip())
        if dernier['contexte'] is None or not choisie:
            self.repondre_json({'erreur': 'correction vide'}, 400)
            return

        enregistree = enregistrer_preference(dernier['contexte'], choisie, dernier['reponse'])
        if enregistree and memoire:
            memoire[-1]['assistant'] = choisie
            sauvegarder_memoire(memoire)
            dernier['reponse'] = choisie
            print(f"préférence enregistrée : {choisie[:50]!r}")

        self.repondre_json({'enregistree': enregistree, 'inconnus': inconnus,
                            'memoire': memoire, 'fenetre': etat_fenetre(),
                            'preferences': compter_preferences()})

unite = 'caractères' if len(tok.morceaux) == len(tok.alphabet) else 'morceaux'
print(f"Modèle {fichier_modele} chargé sur {device}, fenêtre de {taille_contexte} {unite}")
print("Recherche : " + (f"mode {recherche.mode}, {recherche.nb_passages:,} passages" if recherche
                         else "pas d'index (lancer python recherche.py)"))
print(f"{len(memoire)} échanges en mémoire, {compter_preferences()} préférences enregistrées")
print(f"\n  Interface : http://{HOTE}:{PORT}\n")
ThreadingHTTPServer((HOTE, PORT), Interface).serve_forever()
