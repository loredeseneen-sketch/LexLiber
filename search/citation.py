"""Citation gate — CRITICO (vedi CLAUDE.md, regola CITATION).

Nessuna risposta del sistema può uscire senza almeno una fonte con
score >= CITATION_MIN_SCORE. Questo modulo è l'unico punto autorizzato a
decidere se ci sono fonti sufficienti; sia `chat/engine.py` che
`api/routes.py` devono passare sempre da qui prima di restituire una
risposta all'utente.
"""
from __future__ import annotations

from dataclasses import dataclass

from config import settings
from search.retriever import RetrievedPassage

NO_SOURCES_MESSAGE = (
    "Non ho trovato fonti sufficienti per rispondere con certezza."
)


@dataclass
class Citation:
    """Una citazione pronta per essere mostrata all'utente."""

    source: str
    identifier: str
    url: str
    excerpt: str
    score: float
    codice: str = ""
    articolo: str = ""
    rerank_score: float | None = None

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "identifier": self.identifier,
            "url": self.url,
            "excerpt": self.excerpt,
            "score": round(self.score, 4),
        }

    @property
    def label(self) -> str:
        """Etichetta breve per la citazione inline nel testo del LLM, es.
        'Art. 1453, Codice Civile'. Ricade su `identifier` se articolo/codice
        non sono disponibili (es. citazioni costruite manualmente nei test)."""
        if self.articolo and self.codice:
            return f"{self.articolo}, {self.codice}"
        return self.identifier


@dataclass
class CitationGateResult:
    """Esito del controllo citation: o ci sono fonti sufficienti, o no."""

    sufficient: bool
    citations: list[Citation]
    fallback_message: str | None = None


def build_excerpt(text: str, max_lines: int | None = None) -> str:
    max_lines = max_lines or settings.citation_max_excerpt_lines
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        lines = [text.strip()]
    return "\n".join(lines[:max_lines])


def apply_citation_gate(
    passages: list[RetrievedPassage],
    min_score: float | None = None,
) -> CitationGateResult:
    """Filtra i passaggi recuperati applicando la soglia di score minima.

    Un passaggio qualifica se ALMENO UNO tra il cosine score (bi-encoder) e
    il rerank score (cross-encoder, se disponibile) supera la soglia. I due
    modelli sbagliano in modo diverso e non sono comparabili in valore
    assoluto tra query diverse (il cross-encoder è spesso decisivo per
    correggere match lessicali fuorvianti, ma la sua confidenza assoluta
    varia molto a seconda di quanto la domanda è generica — usarlo da solo
    come soglia rigetta erroneamente fonti che il cosine score aveva già
    correttamente qualificato). Vedi search/retriever.py.

    Se nessun passaggio supera la soglia, il gate impone il messaggio di
    fallback fisso: mai inventare una risposta senza fonte (CLAUDE.md).
    """
    min_score = min_score if min_score is not None else settings.citation_min_score

    qualifying = [
        p
        for p in passages
        if p.score >= min_score or (p.rerank_score is not None and p.rerank_score >= min_score)
    ]

    if not qualifying:
        return CitationGateResult(
            sufficient=False, citations=[], fallback_message=NO_SOURCES_MESSAGE
        )

    citations = [
        Citation(
            source=p.source,
            identifier=p.identifier,
            url=p.url,
            excerpt=build_excerpt(p.text),
            score=p.score,
            codice=p.codice,
            articolo=p.articolo,
            rerank_score=p.rerank_score,
        )
        for p in qualifying
    ]
    return CitationGateResult(sufficient=True, citations=citations)


def format_citation(citation: Citation) -> str:
    """Formatta una citazione in testo leggibile per l'utente finale.

    Formato: "Art. X [Nome Codice]: 'estratto' — fonte: url normattiva/eur-lex"
    """
    label = f"{citation.articolo} [{citation.codice}]" if citation.codice else citation.identifier
    return f"{label}: '{citation.excerpt}' — fonte: {citation.url}"
