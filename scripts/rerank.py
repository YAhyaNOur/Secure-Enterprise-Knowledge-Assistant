"""
Module de reranking (cross-encodeur) a brancher dans la recherche finale.

Contrat d'interface :
    rerank(question, candidates, model_path=None, top_n=None) -> list[dict]

- question   : str, la question de l'utilisateur (sans prefixe).
- candidates : list[dict], passages renvoyes par la recherche Qdrant (apres filtre de role).
               Chaque dict doit contenir la cle "text" ou "ptext" :
               "ptext" = "<titre du document> - <titre de section>\n<texte>" (meme forme que
               embed_text de 5-index.py). Si "ptext" absent, "text" est utilise.
- model_path : None = reranker pre-entraine (MiniLM) ; chemin vers models/<nom>/ = reranker affine.
- top_n      : None = renvoie tous les candidats tries ; sinon les top_n premiers.

Retour : les memes dicts, tries par score decroissant, avec une cle supplementaire "rerank_score".

Regles :
- Ne jamais elargir les candidats ici : le reranker ne fait que reordonner ce qui a deja
  passe le filtre de role dans Qdrant. Il ne doit pas voir de passage interdit.
- Le modele est charge une seule fois (cache) pour eviter de recharger a chaque question.
"""

from functools import lru_cache

import numpy as np
from sentence_transformers import CrossEncoder

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_LENGTH = 256  # meme valeur que dans 7-finetune_reranker.py pour rester comparable


@lru_cache(maxsize=4)
def _load(model_path: str | None) -> CrossEncoder:
    return CrossEncoder(model_path or DEFAULT_MODEL, max_length=MAX_LENGTH)


def rerank(question: str, candidates: list[dict], model_path: str | None = None,
           top_n: int | None = None) -> list[dict]:
    if not candidates:
        return []

    ce = _load(model_path)
    texts = [c.get("ptext", c.get("text", "")) for c in candidates]
    scores = np.asarray(ce.predict([(question, t) for t in texts], show_progress_bar=False))

    order = np.argsort(-scores)
    ranked = []
    for i in order:
        item = dict(candidates[i])
        item["rerank_score"] = float(scores[i])
        ranked.append(item)

    return ranked[:top_n] if top_n else ranked
