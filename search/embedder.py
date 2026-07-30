"""Embedding locale via sentence-transformers.

Nessuna chiamata API esterna: il modello viene scaricato una volta (o
incluso nell'immagine Docker) e gira su CPU/GPU locale. Questo è il
componente che garantisce che la ricerca semantica non dipenda da alcun
servizio cloud (principio SOVEREIGN, vedi CLAUDE.md).

Modello di default: paraphrase-multilingual-mpnet-base-v2, un modello
sentence-transformers "puro" (nessun prefisso query:/passage: richiesto,
a differenza della famiglia E5).
"""
from __future__ import annotations

import logging
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from config import settings

logger = logging.getLogger(__name__)


class Embedder:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedding_model
        self._model = _load_model(self.model_name)

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(text, normalize_embeddings=True).tolist()

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False
        )
        return vectors.tolist()

    def count_tokens(self, texts: list[str]) -> int:
        """Conta i token effettivamente processati dal tokenizer del
        modello (esclude il padding), usato per il logging di ingest.py."""
        if not texts:
            return 0
        features = self._model.tokenize(texts)
        attention_mask = features.get("attention_mask")
        if attention_mask is not None:
            return int(attention_mask.sum().item())
        return sum(len(ids) for ids in features["input_ids"])

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()


@lru_cache(maxsize=2)
def _load_model(model_name: str) -> SentenceTransformer:
    logger.info("Caricamento modello di embedding locale: %s", model_name)
    return SentenceTransformer(model_name)
