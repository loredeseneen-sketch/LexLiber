"""API FastAPI di LexLiber.

Espone un unico endpoint funzionale (`POST /api/ask`) usato dal frontend
vanilla JS. Nessun endpoint qui contatta servizi esterni oltre
all'eventuale chiamata a Claude API dentro `chat/engine.py` (disattivabile
con LLM_PROVIDER=ollama). Vedi CLAUDE.md.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from chat.engine import ChatAnswer, ChatEngine
from config import settings
from corpus.meta import read_last_update
from search.document_extractor import EmptyDocumentError, UnsupportedFileTypeError, extract_text
from search.retriever import build_retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="LexLiber", description="Ricerca giuridica italiana sovereign")

# Il frontend è servito da un container nginx separato (vedi docker-compose.yml),
# quindi serve CORS aperto sull'API locale. Nessun dominio esterno è coinvolto.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_engine: ChatEngine | None = None


def get_engine() -> ChatEngine:
    global _engine
    if _engine is None:
        # build_retriever ricade automaticamente sul Qdrant locale embedded
        # se quello configurato (di norma il container Docker) non è
        # raggiungibile, così l'API resta utilizzabile anche senza Docker.
        _engine = ChatEngine(retriever=build_retriever())
    return _engine


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)


class CitationResponse(BaseModel):
    source: str
    identifier: str
    url: str
    excerpt: str
    score: float


class AskResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]
    sufficient_sources: bool


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    try:
        result = get_engine().ask(payload.question)
    except Exception:
        logger.exception("Errore durante la generazione della risposta")
        raise HTTPException(status_code=500, detail="Errore interno durante la ricerca.")

    return AskResponse(
        answer=result.answer,
        citations=[CitationResponse(**c.to_dict()) for c in result.citations],
        sufficient_sources=result.sufficient_sources,
    )


# --- Endpoint per il frontend (frontend/index.html) -------------------------
# Formato di risposta più adatto alla UI (etichetta pronta, confidence a tre
# livelli, tempo di risposta) rispetto ad /api/ask, mantenuto sopra per
# retrocompatibilità con i test/client esistenti.


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)


class FonteItem(BaseModel):
    label: str
    source: str
    url: str
    excerpt: str
    score: float


class SearchResponse(BaseModel):
    risposta: str
    fonti: list[FonteItem]
    confidence: str
    tempo_ms: int


class HealthResponse(BaseModel):
    status: str
    chunk_totali: int
    ultimo_aggiornamento: Optional[str]


def _confidence_score(citation) -> float:
    """Il rerank score (cross-encoder) è un giudizio di rilevanza più
    accurato del cosine score puro (vedi search/retriever.py per i casi che
    hanno motivato l'introduzione del reranker), quindi va preferito per il
    badge mostrato all'utente quando disponibile. Ricade sul cosine score
    solo se il reranker non era attivo per questa ricerca."""
    return citation.rerank_score if citation.rerank_score is not None else citation.score


def _compute_confidence(result: ChatAnswer) -> str:
    """Confidence a tre livelli derivata dagli score delle fonti: il
    citation gate già garantisce score >= CITATION_MIN_SCORE (su almeno uno
    dei due segnali) per ogni fonte restituita, quindi "bassa" corrisponde
    sempre a fonti insufficienti."""
    if not result.sufficient_sources or not result.citations:
        return "bassa"
    top_score = max(_confidence_score(c) for c in result.citations)
    return "alta" if top_score >= 0.85 else "media"


@app.post("/search", response_model=SearchResponse)
def search(payload: SearchRequest) -> SearchResponse:
    start = time.perf_counter()
    try:
        result = get_engine().ask(payload.query)
    except Exception:
        logger.exception("Errore durante la generazione della risposta")
        raise HTTPException(status_code=500, detail="Errore interno durante la ricerca.")

    tempo_ms = round((time.perf_counter() - start) * 1000)
    return SearchResponse(
        risposta=result.answer,
        fonti=[
            FonteItem(
                label=c.label,
                source=c.source,
                url=c.url,
                excerpt=c.excerpt,
                score=round(c.score, 4),
            )
            for c in result.citations
        ],
        confidence=_compute_confidence(result),
        tempo_ms=tempo_ms,
    )


# --- Analisi documento caricato (sezione "Analizza il mio documento") ------
# L'avvocato carica un PDF/DOCX e fa una domanda su quel documento
# specifico: il testo viene estratto localmente (search/document_extractor.py,
# nessuna chiamata esterna) e passato a ChatEngine.analyze_document, che
# combina il documento con le norme del corpus rilevanti per la domanda.


class DocumentAnalysisResponse(BaseModel):
    risposta: str
    fonti_normative: list[FonteItem]
    documento: str
    confidence: str
    tempo_ms: int


@app.post("/analyze-document", response_model=DocumentAnalysisResponse)
async def analyze_document(
    question: str = Form(..., min_length=3, max_length=2000),
    file: UploadFile = File(...),
) -> DocumentAnalysisResponse:
    start = time.perf_counter()
    filename = file.filename or "documento"

    content = await file.read()
    max_bytes = settings.document_max_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File troppo grande (max {settings.document_max_size_mb}MB).",
        )

    try:
        document_text = extract_text(filename, content)
    except UnsupportedFileTypeError:
        raise HTTPException(
            status_code=400, detail="Formato non supportato. Carica un file PDF o DOCX."
        )
    except EmptyDocumentError:
        raise HTTPException(
            status_code=400,
            detail="Nessun testo estraibile dal documento (es. scansione senza OCR).",
        )
    except Exception:
        logger.exception("Errore durante l'estrazione del testo dal documento (%s)", filename)
        raise HTTPException(status_code=400, detail="Impossibile leggere il documento caricato.")

    try:
        result = get_engine().analyze_document(question, document_text)
    except Exception:
        logger.exception("Errore durante l'analisi del documento (%s)", filename)
        raise HTTPException(status_code=500, detail="Errore interno durante l'analisi del documento.")

    tempo_ms = round((time.perf_counter() - start) * 1000)
    return DocumentAnalysisResponse(
        risposta=result.answer,
        fonti_normative=[
            FonteItem(
                label=c.label,
                source=c.source,
                url=c.url,
                excerpt=c.excerpt,
                score=round(c.score, 4),
            )
            for c in result.citations
        ],
        documento=filename,
        confidence=_compute_confidence(result),
        tempo_ms=tempo_ms,
    )


@app.get("/health", response_model=HealthResponse)
def health_full() -> HealthResponse:
    chunk_totali = 0
    try:
        client = get_engine().retriever.client
        chunk_totali = client.count(collection_name=settings.qdrant_collection, exact=True).count
    except Exception:
        logger.warning("Impossibile leggere il conteggio chunk da Qdrant per /health")

    return HealthResponse(
        status="ok",
        chunk_totali=chunk_totali,
        ultimo_aggiornamento=read_last_update(),
    )
