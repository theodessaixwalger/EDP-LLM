import ast
import datetime
import operator
import re


# --- Calculatrice ---
NOMBRES = {'zéro': 0, 'zero': 0, 'un': 1, 'une': 1, 'deux': 2, 'trois': 3, 'quatre': 4,
           'cinq': 5, 'six': 6, 'sept': 7, 'huit': 8, 'neuf': 9, 'dix': 10, 'onze': 11,
           'douze': 12, 'treize': 13, 'quatorze': 14, 'quinze': 15, 'seize': 16,
           'vingt': 20, 'vingts': 20, 'trente': 30, 'quarante': 40, 'cinquante': 50,
           'soixante': 60, 'cent': 100, 'cents': 100, 'mille': 1000}
MOT_NOMBRE = '|'.join(sorted(NOMBRES, key=len, reverse=True))
SEPARATEUR = r"(?:\s+|-)"
NOMBRE_EN_LETTRES = re.compile(rf"\b(?:{MOT_NOMBRE})(?:{SEPARATEUR}(?:et{SEPARATEUR})?(?:{MOT_NOMBRE}))*\b")
OPERATIONS_EN_LETTRES = [
    (re.compile(r"\b(?:multipliée?s? par|multiplie par|fois)\b"), ' * '),
    (re.compile(r"\b(?:divisée?s? par|divise par)\b"), ' / '),
    (re.compile(r"\bplus\b"), ' + '),
    (re.compile(r"\bmoins\b"), ' - '),
    (re.compile(r"\bau carré\b"), ' ** 2 '),
    (re.compile(r"\bau cube\b"), ' ** 3 '),
    (re.compile(r"\b(?:puissance|exposant)\b"), ' ** '),
    (re.compile(r"(?<=\d)\s*[x×]\s*(?=\d)"), ' * '),
    (re.compile(r"÷"), ' / '),
    (re.compile(r"\^"), ' ** '),
]
POURCENTAGE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|pour ?cents?) de\s*")
NOMBRE = r"\d+(?:\.\d+)?"
EXPRESSION = re.compile(rf"[(\s-]*{NOMBRE}\)*(?:\s*(?:\*\*|[-+*/])\s*[(\s-]*{NOMBRE}\)*)+")
DEMANDE_DE_CALCUL = re.compile(r"\b(?:combien|calcul\w*|résultat|font|fait|égale?s?|vaut)\b|=")
SEULEMENT_UN_CALCUL = re.compile(r"[\d\s.()+\-*/=?!]+")
OPERATEURS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
              ast.Div: operator.truediv, ast.Pow: operator.pow}

def valeur(nombre_en_lettres):
    total, courant = 0, 0
    for mot in re.split(r"[\s-]+", nombre_en_lettres):
        if mot == 'et':
            continue
        if mot in ('cent', 'cents'):
            courant = (courant or 1) * 100
        elif mot == 'mille':
            total += (courant or 1) * 1000
            courant = 0
        elif mot in ('vingt', 'vingts') and courant % 100 == 4:
            courant += 76
        else:
            courant += NOMBRES[mot]
    return total + courant

# Jamais de eval() : on parcourt l'arbre et on accepte que des nombres et des opérations,
# comme ça un message du genre __import__('os') fait rien
def evaluer(noeud):
    if isinstance(noeud, ast.Expression):
        return evaluer(noeud.body)
    if isinstance(noeud, ast.Constant) and type(noeud.value) in (int, float):
        return noeud.value
    if isinstance(noeud, ast.UnaryOp) and isinstance(noeud.op, (ast.USub, ast.UAdd)):
        return -evaluer(noeud.operand) if isinstance(noeud.op, ast.USub) else evaluer(noeud.operand)
    if isinstance(noeud, ast.BinOp) and type(noeud.op) in OPERATEURS:
        gauche, droite = evaluer(noeud.left), evaluer(noeud.right)
        if isinstance(noeud.op, ast.Pow) and (abs(droite) > 100 or abs(gauche) > 1e6):
            raise ValueError("puissance trop grande")
        return OPERATEURS[type(noeud.op)](gauche, droite)
    raise ValueError("pas un calcul")

def ecrire_nombre(x):
    if float(x).is_integer() and abs(x) < 1e15:
        return f"{int(x):,}".replace(',', ' ')
    return f"{x:.6g}".replace('.', ',')

def calcul(message):
    original = message.lower().replace('’', "'")
    texte = NOMBRE_EN_LETTRES.sub(lambda m: str(valeur(m.group())), original)
    texte = re.sub(r"(?<=\d),(?=\d)", '.', texte)
    for motif, symbole in OPERATIONS_EN_LETTRES:
        texte = motif.sub(symbole, texte)
    texte = POURCENTAGE.sub(r"\1 / 100 * ", texte)
    seulement_un_calcul = SEULEMENT_UN_CALCUL.fullmatch(texte)
    if not seulement_un_calcul and not DEMANDE_DE_CALCUL.search(original):
        return None
    # « la guerre de 1914-1918 » c'est pas une soustraction
    if not seulement_un_calcul:
        texte = re.sub(r"(?<=\d)-(?=\d)", ' à ', texte)
    trouve = EXPRESSION.search(texte)
    if not trouve:
        return None
    expression = ' '.join(trouve.group().split())
    try:
        resultat = evaluer(ast.parse(expression, mode='eval'))
    except ZeroDivisionError:
        return "On ne peut pas diviser par zéro."
    except (SyntaxError, ValueError, OverflowError):
        return None
    lisible = re.sub(r"\s*\*\*\s*", ' puissance ', expression)
    lisible = re.sub(r"(?<=[\d)])\s*([-+*/])\s*", r" \1 ", lisible).replace('*', '×').replace('.', ',')
    lisible = re.sub(r" / 100 × ", ' % de ', lisible)
    return f"{lisible} = {ecrire_nombre(resultat)}."

# --- Horloge ---
JOURS = ['lundi', 'mardi', 'mercredi', 'jeudi', 'vendredi', 'samedi', 'dimanche']
MOIS = ['janvier', 'février', 'mars', 'avril', 'mai', 'juin', 'juillet', 'août',
        'septembre', 'octobre', 'novembre', 'décembre']
# Tournures précises pour pas répondre l'heure à « à quelle heure ouvre le Louvre ? »
QUESTION_DATE = re.compile(r"quel jour (?:on est|sommes-nous|est-on|nous sommes|est-ce|aujourd'hui)|"
                           r"on est quel jour|on est le combien|quelle (?:est la )?date (?:on est|sommes-nous|aujourd'hui|d'aujourd'hui)|"
                           r"la date d'aujourd'hui|(?:en )?quelle année (?:on est|sommes-nous)|on est en quelle année")
QUESTION_HEURE = re.compile(r"quelle heure (?:est-il|il est)|il est quelle heure|l'heure qu'il est|quelle heure on est")

def horloge(message, maintenant=None):
    texte = message.lower().replace('’', "'")
    maintenant = maintenant or datetime.datetime.now()
    if QUESTION_HEURE.search(texte):
        return f"Il est {maintenant.hour} h {maintenant.minute:02d}."
    if QUESTION_DATE.search(texte):
        return (f"Nous sommes le {JOURS[maintenant.weekday()]} {maintenant.day} "
                f"{MOIS[maintenant.month - 1]} {maintenant.year}.")
    return None

def utiliser(message):
    for nom, outil in (('Calculatrice', calcul), ('Horloge', horloge)):
        resultat = outil(message)
        if resultat:
            return nom, resultat
    return None

if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    print(utiliser(' '.join(sys.argv[1:])))
