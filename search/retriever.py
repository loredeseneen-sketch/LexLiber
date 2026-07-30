"""Retriever su Qdrant locale (Docker, nessun account cloud).

Il retriever è l'unico componente che parla con Qdrant: incapsula
creazione della collezione, upsert dei chunk indicizzati offline
(vedi corpus/ingest.py) e ricerca semantica a runtime.

Il payload salvato su Qdrant segue lo schema `ArticleChunk`
(fonte/codice/libro/titolo/capo/articolo/numero_articolo/testo/
url_originale, vedi corpus/models.py). `RetrievedPassage` è invece
l'interfaccia stabile usata da chat/citation/api, così i livelli a valle
non dipendono dai dettagli del payload.

NOTA per una sessione futura: `LegalChunk` (schema usato da
corpus/scrapers/cassazione.py, non ancora ricollegato a ingest.py) ha una
forma diversa e non è ancora gestito da questo retriever.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from config import settings
from corpus.models import ArticleChunk
from search.embedder import Embedder
from search.reranker import Reranker

logger = logging.getLogger(__name__)

LOCAL_QDRANT_PATH = "./data/qdrant_local"

ARTICLE_NOT_FOUND_MESSAGE = (
    "Articolo non trovato nel corpus. Potrebbe essere stato abrogato o non ancora indicizzato."
)

# Mappa gruppo regex nominato -> valore esatto del campo `codice` salvato su
# Qdrant (vedi corpus/scrapers/normattiva.py:CODICI). Gli alias più lunghi/
# specifici (cpc, cpp) sono elencati PRIMA di quelli più corti (cc, cp) negli
# alternation group qui sotto, così il backtracking di `re` non fa mai
# matchare "cpc"/"cpp" come se fossero "cc"/"cp" con una lettera avanzata.
_CODICE_BY_GROUP = {
    "cc": "Codice Civile",
    "cpc": "Codice di Procedura Civile",
    "cp": "Codice Penale",
    "cpp": "Codice di Procedura Penale",
}

# Riconosce riferimenti espliciti a un articolo con indicazione del codice,
# nelle forme "art. N cc", "articolo N codice civile", "N c.c.", ecc.
# Il numero può avere un suffisso letterale separato da spazio (es. "2409
# quinquiesdecies", coerente con come è salvato numero_articolo in Qdrant).
# L'indicatore di codice è SEMPRE obbligatorio (mai un numero nudo): senza
# di esso numero_articolo da solo è ambiguo tra più codici (vedi analisi
# diagnostica: art. 27 e art. 743 esistono in codici diversi con contenuto
# diverso), quindi un bare number non deve mai attivare il lookup esatto.
_ARTICLE_QUERY_RE = re.compile(
    r"""
    (?:art\.?|articolo)?\s*          # prefisso opzionale "art." / "articolo"
    (?P<numero>\d+)                  # numero articolo
    (?:\s+(?P<suffisso>[a-z]+))?     # eventuale suffisso letterale (es. "quinquiesdecies")
    \s*
    (?:
        (?P<cpc>c\.?\s*p\.?\s*c\.?|codice\s+di\s+procedura\s+civile)
        |(?P<cpp>c\.?\s*p\.?\s*p\.?|codice\s+di\s+procedura\s+penale)
        |(?P<cc>c\.?\s*c\.?|codice\s+civile)
        |(?P<cp>c\.?\s*p\.?|codice\s+penale)
    )
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


def detect_article_query(query: str) -> tuple[str, str] | None:
    """Riconosce riferimenti diretti a un articolo (es. "art. 1418 cc",
    "articolo 743 codice civile", "1337 cc", "art 27 cp") e restituisce
    `(numero_articolo, codice)` pronto per il lookup esatto su Qdrant, o
    `None` se la query non è un riferimento articolo strutturato (in tal
    caso il chiamante deve ricadere sul retriever semantico invariato)."""
    match = _ARTICLE_QUERY_RE.match(query.strip())
    if not match:
        return None

    numero = match.group("numero")
    if match.group("suffisso"):
        numero = f"{numero} {match.group('suffisso').lower()}"

    for group_name, codice in _CODICE_BY_GROUP.items():
        if match.group(group_name):
            return numero, codice
    return None


@dataclass
class RetrievedPassage:
    """Un passaggio recuperato da Qdrant, con score di similarità.

    `score` resta sempre il cosine score del bi-encoder (familiare, sempre
    popolato, sulla stessa scala 0-1 di CITATION_MIN_SCORE). `rerank_score`
    è invece la confidenza del cross-encoder (sigmoide del logit grezzo,
    anch'essa 0-1) quando il reranking è attivo, altrimenti None. Il gate
    di citazione (search/citation.py) qualifica un passaggio se ALMENO UNO
    dei due segnali supera la soglia: i due modelli commettono errori
    diversi (vedi search/reranker.py e retriever.search per esempi
    concreti), quindi combinarli in OR massimizza il recall senza allentare
    la soglia in modo indiscriminato."""

    score: float
    doc_id: str
    source: str
    title: str
    identifier: str
    url: str
    text: str
    date: str | None = None
    codice: str = ""
    articolo: str = ""
    rerank_score: float | None = None

    @property
    def excerpt(self, max_lines: int = None) -> str:  # noqa: RUF013
        max_lines = max_lines or settings.citation_max_excerpt_lines
        lines = [ln for ln in self.text.splitlines() if ln.strip()]
        if not lines:
            lines = [self.text]
        return "\n".join(lines[:max_lines])


class Retriever:
    def __init__(
        self,
        embedder: Embedder | None = None,
        client: QdrantClient | None = None,
        reranker: Reranker | None = None,
    ):
        self.embedder = embedder or Embedder()
        self.client = client or QdrantClient(
            host=settings.qdrant_host, port=settings.qdrant_port
        )
        self.collection = settings.qdrant_collection
        self.reranker = reranker if reranker is not None else (
            Reranker() if settings.rerank_enabled else None
        )

    def ensure_collection(self) -> None:
        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection in existing:
            return
        logger.info("Creo collezione Qdrant %r", self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(
                size=self.embedder.dimension, distance=qm.Distance.COSINE
            ),
        )

    def upsert_chunks(self, chunks: list[ArticleChunk]) -> None:
        if not chunks:
            return
        # La rubrica (es. "Art. 1337. (Trattative e responsabilità
        # precontrattuale).") spesso contiene il termine giuridico esatto
        # cercato dall'utente, ma viene salvata separatamente da `testo`
        # nel payload (per non sporcare l'estratto mostrato in citazione).
        # Includerla nel testo effettivamente incorporato migliora
        # sensibilmente il recall semantico per query che usano la
        # terminologia della rubrica invece che del corpo dell'articolo.
        vectors = self.embedder.embed_passages(
            [f"{c.articolo} {c.testo}" for c in chunks]
        )
        points = [
            qm.PointStruct(id=_stable_int_id(c.chunk_id), vector=vec, payload=c.to_payload())
            for c, vec in zip(chunks, vectors)
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedPassage]:
        top_k = top_k or settings.citation_top_k
        query_vector = self.embedder.embed_query(query)

        # Se il reranking è attivo recuperiamo un pool di candidati più
        # ampio del top-k finale: il cosine score del bi-encoder da solo
        # scarta erroneamente articoli concettualmente centrali (vedi
        # search/reranker.py per un esempio concreto riscontrato in test).
        candidate_k = (
            max(top_k * settings.rerank_candidate_multiplier, settings.rerank_min_candidates)
            if self.reranker
            else top_k
        )
        hits = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=candidate_k,
            with_payload=True,
        )
        passages = [_passage_from_hit(hit) for hit in hits]

        if self.reranker is None or not passages:
            return passages[:top_k]

        # Rerankiamo l'INTERO pool di candidati, non solo quelli che già
        # superano la soglia di citation sul cosine score. Prefiltrare per
        # cosine prima del reranking è il bug diagnosticato manualmente:
        # per query brevi e generiche il bi-encoder può assegnare uno score
        # basso ad articoli concettualmente centrali (es. "Quando è nullo un
        # contratto?" → Art. 1418 c.c. aveva cosine 0.67, sotto soglia 0.75,
        # pur essendo giudicato nettamente il più rilevante dal cross-encoder,
        # mentre un articolo marginale con cosine 0.77 restava l'unico
        # candidato passato al gate). Il cross-encoder deve vedere l'intero
        # pool per poter correggere questi errori del bi-encoder.
        rerank_scores = self.reranker.rerank(
            query, [f"{p.articolo} {p.text}" for p in passages]
        )
        if rerank_scores is None:
            return passages[:top_k]

        # Il cross-encoder restituisce logit non normalizzati (possono
        # essere negativi o >1): applichiamo una sigmoide per riportarli su
        # una scala 0-1, salvata in `rerank_score` accanto al cosine score
        # originale (mai sovrascritto, vedi RetrievedPassage). Usare la
        # confidenza del reranker (non il cosine) per l'ORDINAMENTO finale è
        # ciò che corregge casi come "termini per esercitare i rimedi da
        # inadempimento": l'Art. 1184 (Termine, un semplice omonimo lessicale
        # su "scadenza del pagamento") ha cosine più alto degli articoli
        # davvero pertinenti (1453, 1218) ma il cross-encoder lo giudica
        # meno rilevante e lo retrocede correttamente.
        for p, raw_score in zip(passages, rerank_scores):
            p.rerank_score = _sigmoid(raw_score)

        passages.sort(key=lambda p: p.rerank_score, reverse=True)
        return passages[:top_k]

    def search_by_article(self, numero_articolo: str, codice: str) -> RetrievedPassage | None:
        """Lookup ESATTO (non semantico) su Qdrant: filtra per `numero_articolo`
        e `codice` simultaneamente sul payload, senza calcolare alcun
        embedding né applicare soglie di similarità. `numero_articolo` da
        solo è ambiguo tra codici diversi (es. l'art. 27 esiste sia nel
        Codice Civile che nel Codice Penale con contenuto diverso), quindi
        entrambi i filtri sono obbligatori per un match sicuro. Usato da
        `detect_article_query` quando la query è un riferimento articolo
        strutturato (vedi chat/engine.py)."""
        points, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=qm.Filter(
                must=[
                    qm.FieldCondition(
                        key="numero_articolo", match=qm.MatchValue(value=numero_articolo)
                    ),
                    qm.FieldCondition(key="codice", match=qm.MatchValue(value=codice)),
                ]
            ),
            limit=1,
            with_payload=True,
        )
        if not points:
            return None
        # score=1.0/rerank_score=1.0: è un lookup esatto per metadata, non un
        # match di similarità, ma riusa senza modifiche il citation gate
        # esistente (search/citation.py), che qualifica qualunque passaggio
        # con score >= soglia — 1.0 lo supera sempre, senza bisogno di un
        # bypass ad-hoc del gate.
        return _passage_from_payload(points[0].payload, score=1.0, rerank_score=1.0)


def build_retriever(embedder: Embedder | None = None) -> Retriever:
    """Si connette al Qdrant configurato (di norma quello Docker di
    docker-compose.yml, tramite QDRANT_HOST/QDRANT_PORT); se non è
    raggiungibile (es. Docker Desktop non installato/avviato) ricade su
    un'istanza Qdrant locale embedded persistita su disco (LOCAL_QDRANT_PATH),
    così sia l'ingestione (corpus/ingest.py) sia l'API a runtime
    (api/routes.py) restano utilizzabili end-to-end senza Docker. Un solo
    punto di fallback condiviso evita che i due caller divergano."""
    embedder = embedder or Embedder()
    retriever = Retriever(embedder=embedder)
    try:
        retriever.client.get_collections()
        return retriever
    except Exception:
        logger.warning(
            "Qdrant non raggiungibile su %s:%s (Docker non avviato?): uso il client "
            "Qdrant locale embedded su %s.",
            settings.qdrant_host,
            settings.qdrant_port,
            LOCAL_QDRANT_PATH,
        )
        return Retriever(embedder=embedder, client=QdrantClient(path=LOCAL_QDRANT_PATH))


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _passage_from_hit(hit) -> RetrievedPassage:
    return _passage_from_payload(hit.payload, score=hit.score)


def _passage_from_payload(
    payload: dict, score: float, rerank_score: float | None = None
) -> RetrievedPassage:
    return RetrievedPassage(
        score=score,
        doc_id=payload.get("chunk_id", ""),
        source=payload["fonte"],
        title=f'{payload["codice"]} — {payload["articolo"]}',
        identifier=_build_identifier(payload),
        url=payload["url_originale"],
        text=payload["testo"],
        date=None,
        codice=payload.get("codice", ""),
        articolo=payload.get("articolo", ""),
        rerank_score=rerank_score,
    )


def _build_identifier(payload: dict) -> str:
    """Costruisce una citazione leggibile tipo:
    'Codice Civile, Libro IV, Titolo II, Capo XIV, Art. 1453'."""
    parti = [payload["codice"]]
    for campo in ("libro", "titolo", "capo"):
        valore = payload.get(campo)
        if valore:
            parti.append(valore)
    parti.append(payload["articolo"])
    return ", ".join(parti)


def _stable_int_id(chunk_id: str) -> int:
    """Qdrant richiede id numerici o UUID: deriviamo un intero stabile."""
    import hashlib

    return int(hashlib.sha256(chunk_id.encode("utf-8")).hexdigest()[:16], 16)
