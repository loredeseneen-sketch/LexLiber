"""Modelli dati condivisi tra scraper, ingest e retriever.

Due schemi convivono in questo modulo:

- `LegalDocument`/`LegalChunk`: schema generico usato dallo scraper
  Cassazione (non ancora ricollegato a `corpus/ingest.py`, previsto per
  una sessione successiva).
- `ArticleChunk`: schema strutturato "un articolo = un chunk" usato dagli
  scraper Normattiva ed EUR-Lex (Sessione 1), con i metadati richiesti da
  CLAUDE.md: fonte, codice, libro, titolo, capo, articolo, numero_articolo,
  testo, url_originale.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field


@dataclass
class LegalDocument:
    """Un documento giuridico normalizzato, indipendente dalla fonte."""

    source: str          # "normattiva" | "eurlex" | "cassazione"
    title: str           # es. "Codice Civile, art. 1372"
    identifier: str      # es. "Regio Decreto 16 marzo 1942, n. 262, art. 1372"
    url: str             # link pubblico diretto alla fonte
    text: str            # testo integrale della norma/massima
    date: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        raw = f"{self.source}:{self.identifier}:{self.url}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


@dataclass
class LegalChunk:
    """Una porzione di testo di un LegalDocument, pronta per l'embedding."""

    chunk_id: str
    doc_id: str
    source: str
    title: str
    identifier: str
    url: str
    text: str
    date: str | None = None

    def to_payload(self) -> dict:
        """Payload salvato su Qdrant assieme al vettore."""
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "title": self.title,
            "identifier": self.identifier,
            "url": self.url,
            "text": self.text,
            "date": self.date,
        }


@dataclass
class ArticleChunk:
    """Un singolo articolo di un codice/atto normativo: è già il chunk
    indicizzabile, non serve ulteriore suddivisione (un articolo = un
    chunk, come richiesto da CLAUDE.md per Normattiva ed EUR-Lex)."""

    fonte: str            # "normattiva" | "eurlex"
    codice: str           # es. "Codice Civile" oppure il titolo dell'atto UE
    libro: str | None     # es. "LIBRO IV - Delle obbligazioni"
    titolo: str | None    # es. "TITOLO II - Dei contratti in generale"
    capo: str | None      # es. "CAPO XIV - Della risoluzione del contratto"
    articolo: str         # etichetta completa, es. "Art. 1453"
    numero_articolo: str  # solo il numero, es. "1453"
    testo: str            # testo integrale dell'articolo
    url_originale: str    # link diretto alla fonte pubblica

    @property
    def chunk_id(self) -> str:
        raw = f"{self.fonte}:{self.codice}:{self.numero_articolo}:{self.url_originale}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def to_payload(self) -> dict:
        """Payload salvato su Qdrant assieme al vettore: include il
        chunk_id oltre a tutti i campi di metadato richiesti."""
        return asdict(self) | {"chunk_id": self.chunk_id}
