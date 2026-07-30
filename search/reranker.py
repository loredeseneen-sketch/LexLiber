"""Reranking locale via cross-encoder (nessuna chiamata API esterna).

Il retrieval denso (bi-encoder, vedi search/embedder.py) soffre di errori
di ranking su query generiche: articoli concettualmente centrali possono
restare fuori dal top-k perché articoli marginali condividono più lessico
superficiale con la query. Esempio riscontrato in test manuale: per "In
caso di inadempimento contrattuale, quali rimedi ha il creditore e quali
sono i termini per esercitarli?" l'Art. 1453 c.c. (risoluzione per
inadempimento, il rimedio realmente centrale) finiva oltre il rank 20,
scavalcato dall'Art. 1184 "Termine" — che parla di scadenza del pagamento,
non di rimedi contro l'inadempimento, ma condivide la parola "termine".

Un cross-encoder multilingue, che valuta query e passaggio insieme invece
che con embedding separati, corregge questi errori di ranking a un costo
accettabile (poche decine di coppie per richiesta, gestito da
Retriever.search tramite un pool di candidati più ampio del top-k finale).

Se il modello non è disponibile (rete assente, prima esecuzione senza
cache locale) il retriever ricade silenziosamente sull'ordinamento per
similarità coseno: il reranking è un affinamento, non una dipendenza
rigida (principio SEMPLICE/SOVEREIGN, vedi CLAUDE.md).
"""
from __future__ import annotations

import logging
from functools import lru_cache

from config import settings

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.rerank_model
        self._model = _load_model(self.model_name)

    @property
    def available(self) -> bool:
        return self._model is not None

    def rerank(self, query: str, passages: list[str]) -> list[float] | None:
        """Restituisce uno score di rilevanza per ciascun passaggio, nello
        stesso ordine di `passages`. Restituisce None se il modello non è
        disponibile, così il chiamante può ricadere sul cosine score."""
        if self._model is None or not passages:
            return None
        pairs = [(query, p) for p in passages]
        return self._model.predict(pairs).tolist()


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    try:
        from sentence_transformers import CrossEncoder

        logger.info("Caricamento modello di reranking locale: %s", model_name)
        return CrossEncoder(model_name)
    except Exception:
        logger.warning(
            "Reranker %r non disponibile (rete assente o modello non in "
            "cache): ricado sull'ordinamento per similarità coseno.",
            model_name,
        )
        return None
