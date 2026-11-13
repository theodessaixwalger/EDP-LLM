import torch
import torch.nn as nn
import torch.nn.functional as F

# RoPE : on fait tourner q et k selon leur position, comme ça l'attention dépend
# seulement de l'écart entre deux mots et pas de leur place dans la fenêtre
def tourner(x, cos, sin):
    moitie = x.shape[-1] // 2
    x1, x2 = x[..., :moitie], x[..., moitie:]
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)

# anti-boucle : interdit de recopier une suite de n morceaux déjà écrite dans la réponse
def bloquer_repetitions(logits, deja_ecrits, n=16):
    if len(deja_ecrits) < n:
        return logits
    fin = deja_ecrits[-(n - 1):]
    interdits = {deja_ecrits[i + n - 1] for i in range(len(deja_ecrits) - n + 1)
                 if deja_ecrits[i:i + n - 1] == fin}
    if not interdits:
        return logits
    logits = logits.clone()
    logits[list(interdits)] = float('-inf')
    return logits

class AttentionMultiTetes(nn.Module):
    def __init__(self, dim_embed, nb_tetes, dropout):
        super().__init__()
        self.nb_tetes = nb_tetes
        self.requete = nn.Linear(dim_embed, dim_embed, bias=False)
        self.cle = nn.Linear(dim_embed, dim_embed, bias=False)
        self.valeur = nn.Linear(dim_embed, dim_embed, bias=False)
        self.projection = nn.Linear(dim_embed, dim_embed)
        self.dropout_poids = dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cos, sin):
        B, T, C = x.shape

        def par_tete(t):
            return t.view(B, T, self.nb_tetes, C // self.nb_tetes).transpose(1, 2)
        q = tourner(par_tete(self.requete(x)), cos, sin)
        k = tourner(par_tete(self.cle(x)), cos, sin)
        v = par_tete(self.valeur(x))

        # Toutes les têtes en un seul appel, sans jamais stocker la matrice 4096x4096
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout_poids if self.training else 0.0)

        y = y.transpose(1, 2).reshape(B, T, C)
        return self.dropout(self.projection(y))

class ReseauDense(nn.Module):
    def __init__(self, dim_embed, dropout):
        super().__init__()
        self.reseau = nn.Sequential(
            nn.Linear(dim_embed, 4 * dim_embed),
            nn.GELU(),
            nn.Linear(4 * dim_embed, dim_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.reseau(x)

class BlocTransformer(nn.Module):
    def __init__(self, dim_embed, nb_tetes, dropout):
        super().__init__()
        self.attention = AttentionMultiTetes(dim_embed, nb_tetes, dropout)
        self.dense = ReseauDense(dim_embed, dropout)
        self.norme1 = nn.LayerNorm(dim_embed)
        self.norme2 = nn.LayerNorm(dim_embed)

    def forward(self, x, cos, sin):
        x = x + self.attention(self.norme1(x), cos, sin)
        x = x + self.dense(self.norme2(x))
        return x

class MonLLM(nn.Module):
    def __init__(self, taille_vocab, dim_embed=384, nb_tetes=6, nb_blocs=6,
                 taille_contexte=256, dropout=0.2):
        super().__init__()
        self.taille_contexte = taille_contexte
        self.embed_tokens = nn.Embedding(taille_vocab, dim_embed)
        self.blocs = nn.ModuleList([
            BlocTransformer(dim_embed, nb_tetes, dropout)
            for _ in range(nb_blocs)
        ])
        self.norme_finale = nn.LayerNorm(dim_embed)
        self.tete_sortie = nn.Linear(dim_embed, taille_vocab)

        taille_tete = dim_embed // nb_tetes
        vitesses = 1.0 / (10000 ** (torch.arange(0, taille_tete, 2).float() / taille_tete))
        angles = torch.outer(torch.arange(taille_contexte).float(), vitesses)
        self.register_buffer('cos', angles.cos(), persistent=False)
        self.register_buffer('sin', angles.sin(), persistent=False)

        # La fenêtre et le nombre de têtes se lisent pas dans les poids, donc on les sauvegarde avec
        self.register_buffer('fenetre', torch.tensor(taille_contexte))
        self.register_buffer('nb_tetes', torch.tensor(nb_tetes))

    # Retrouve la taille du réseau à partir du fichier de poids (les vieux fichiers avaient une tête tous les 64)
    @staticmethod
    def architecture(etat):
        taille_vocab, dim_embed = etat['embed_tokens.weight'].shape
        nb_blocs = 1 + max(int(cle.split('.')[1]) for cle in etat if cle.startswith('blocs.'))
        nb_tetes = int(etat['nb_tetes']) if 'nb_tetes' in etat else dim_embed // 64
        return dict(taille_vocab=taille_vocab, dim_embed=dim_embed, nb_tetes=nb_tetes,
                    nb_blocs=nb_blocs, taille_contexte=int(etat['fenetre']))

    @classmethod
    def charger(cls, fichier, device):
        etat = torch.load(fichier, map_location=device, weights_only=True)
        modele = cls(**cls.architecture(etat))
        etat.setdefault('nb_tetes', modele.nb_tetes)
        modele.load_state_dict(etat)
        return modele.to(device).eval()

    def forward(self, idx, cibles=None):
        # Bfloat16 si la carte le gère : plus rapide et moitié moins de mémoire, les poids restent en float32
        demi_precision = idx.is_cuda and torch.cuda.is_bf16_supported()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=demi_precision):
            B, T = idx.shape
            x = self.embed_tokens(idx)
            cos, sin = self.cos[:T], self.sin[:T]
            for bloc in self.blocs:
                x = bloc(x, cos, sin)
            x = self.norme_finale(x)
            logits = self.tete_sortie(x)
        logits = logits.float()

        if cibles is None:
            perte = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            cibles = cibles.view(B*T)
            perte = F.cross_entropy(logits, cibles)

        return logits, perte

    @torch.no_grad()
    def generer(self, idx, max_nouveaux_tokens, temperature=1.0):
        for _ in range(max_nouveaux_tokens):
            idx_cond = idx[:, -self.taille_contexte:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_suivant = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_suivant), dim=1)
        return idx
