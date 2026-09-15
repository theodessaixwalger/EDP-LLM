import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TeteAttention(nn.Module):
    """Une tête d'attention : chaque token va chercher de l'information dans les
    tokens qui le précèdent, pondérée par leur pertinence."""
    def __init__(self, dim_embed, taille_tete, taille_contexte, dropout):
        super().__init__()
        # requete = ce que le token cherche, cle = ce qu'il propose aux autres,
        # valeur = l'information qu'il transmet s'il est retenu
        self.cle = nn.Linear(dim_embed, taille_tete, bias=False)
        self.requete = nn.Linear(dim_embed, taille_tete, bias=False)
        self.valeur = nn.Linear(dim_embed, taille_tete, bias=False)
        # Matrice triangulaire enregistrée avec le modèle mais jamais entraînée :
        # elle marque les positions futures, interdites en lecture.
        self.register_buffer('masque', torch.tril(torch.ones(taille_contexte, taille_contexte)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape  # séquences du batch, longueur, dimension d'embedding
        k = self.cle(x)
        q = self.requete(x)
        # Produit scalaire requête/clé = affinité entre chaque paire de tokens.
        # La division par racine(taille_tete) évite des scores trop grands, qui
        # rendraient le softmax quasi binaire et bloqueraient l'apprentissage.
        poids = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        # -inf sur les positions futures : après le softmax leur poids tombe à 0,
        # donc un token ne peut pas tricher en lisant la suite du texte.
        poids = poids.masked_fill(self.masque[:T, :T] == 0, float('-inf'))
        poids = F.softmax(poids, dim=-1)  # affinités transformées en proportions
        poids = self.dropout(poids)
        v = self.valeur(x)
        return poids @ v  # moyenne des valeurs, pondérée par l'attention

class AttentionMultiTetes(nn.Module):
    """Plusieurs têtes en parallèle, chacune libre de se spécialiser sur un type
    de relation différent (accord, ponctuation, mot précédent...)."""
    def __init__(self, dim_embed, nb_tetes, taille_contexte, dropout):
        super().__init__()
        # Les têtes se partagent la dimension d'embedding, donc leurs sorties
        # concaténées retombent exactement sur dim_embed.
        taille_tete = dim_embed // nb_tetes
        self.tetes = nn.ModuleList([
            TeteAttention(dim_embed, taille_tete, taille_contexte, dropout)
            for _ in range(nb_tetes)
        ])
        self.projection = nn.Linear(dim_embed, dim_embed)  # remélange ce que les têtes ont trouvé
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Les têtes sont mises bout à bout sur la dimension des caractéristiques
        sortie = torch.cat([tete(x) for tete in self.tetes], dim=-1)
        return self.dropout(self.projection(sortie))

class ReseauDense(nn.Module):
    """L'attention fait circuler l'information entre tokens ; ce réseau retravaille
    ensuite chaque token isolément."""
    def __init__(self, dim_embed, dropout):
        super().__init__()
        # Passage par une couche cachée 4 fois plus large : cet espace
        # intermédiaire donne au modèle de la place pour ses calculs.
        self.reseau = nn.Sequential(
            nn.Linear(dim_embed, 4 * dim_embed),
            nn.GELU(),
            nn.Linear(4 * dim_embed, dim_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.reseau(x)

class BlocTransformer(nn.Module):
    """Brique de base empilée nb_blocs fois : communication entre tokens, puis
    traitement individuel."""
    def __init__(self, dim_embed, nb_tetes, taille_contexte, dropout):
        super().__init__()
        self.attention = AttentionMultiTetes(dim_embed, nb_tetes, taille_contexte, dropout)
        self.dense = ReseauDense(dim_embed, dropout)
        self.norme1 = nn.LayerNorm(dim_embed)
        self.norme2 = nn.LayerNorm(dim_embed)

    def forward(self, x):
        # "x + ..." : chaque sous-couche ajoute une correction à l'entrée au lieu
        # de la remplacer, ce qui laisse le gradient traverser toute la pile.
        # La normalisation est appliquée avant la sous-couche, pas après : plus
        # stable à l'entraînement.
        x = x + self.attention(self.norme1(x))
        x = x + self.dense(self.norme2(x))
        return x

class MonLLM(nn.Module):
    """Le modèle complet : embeddings, pile de blocs, puis prédiction du
    caractère suivant pour chaque position."""
    def __init__(self, taille_vocab, dim_embed=384, nb_tetes=6, nb_blocs=6,
                 taille_contexte=256, dropout=0.2):
        super().__init__()
        self.taille_contexte = taille_contexte
        # Deux embeddings distincts : l'identité du caractère, et sa position
        # dans la séquence. Sans le second, le modèle verrait un sac de lettres.
        self.embed_tokens = nn.Embedding(taille_vocab, dim_embed)
        self.embed_position = nn.Embedding(taille_contexte, dim_embed)
        self.blocs = nn.Sequential(*[
            BlocTransformer(dim_embed, nb_tetes, taille_contexte, dropout)
            for _ in range(nb_blocs)
        ])
        self.norme_finale = nn.LayerNorm(dim_embed)
        self.tete_sortie = nn.Linear(dim_embed, taille_vocab)  # un score par caractère du vocabulaire

    def forward(self, idx, cibles=None):
        B, T = idx.shape
        emb_tok = self.embed_tokens(idx)
        emb_pos = self.embed_position(torch.arange(T, device=idx.device))
        x = emb_tok + emb_pos  # chaque token porte à la fois son identité et sa position
        x = self.blocs(x)
        x = self.norme_finale(x)
        logits = self.tete_sortie(x)

        # Sans cibles on se contente de prédire (génération). Avec cibles on
        # mesure l'erreur : cross_entropy attend une liste plate d'exemples, d'où
        # l'aplatissement des dimensions batch et temps en une seule.
        if cibles is None:
            perte = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            cibles = cibles.view(B*T)
            perte = F.cross_entropy(logits, cibles)

        return logits, perte

    @torch.no_grad()  # en génération on n'apprend plus : inutile de suivre les gradients
    def generer(self, idx, max_nouveaux_tokens, temperature=1.0):
        # Génération auto-régressive : on prédit un caractère, on l'ajoute au
        # contexte, et on recommence à partir du texte ainsi allongé.
        for _ in range(max_nouveaux_tokens):
            idx_cond = idx[:, -self.taille_contexte:]  # le modèle ne voit jamais plus loin que taille_contexte
            logits, _ = self(idx_cond)
            # Seule la dernière position prédit la suite. La température écrase
            # (< 1, texte plus sage) ou étale (> 1, texte plus varié) les scores.
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            # Tirage au sort selon les probabilités : toujours prendre le
            # caractère le plus probable ferait vite tourner le texte en boucle.
            idx_suivant = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_suivant), dim=1)
        return idx
