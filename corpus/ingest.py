#!/usr/bin/env python3
"""Pipeline di ingestione offline: scraping → embedding → Qdrant.

Uso:
    python corpus/ingest.py --source normattiva
    python corpus/ingest.py --source normattiva --codici codice_civile,dlgs_231_2001 --limit 50
    python corpus/ingest.py --source eurlex --limit 5
    python corpus/ingest.py --source all --dry-run

Questo è l'UNICO punto del sistema in cui vengono fatte chiamate di rete
verso fonti esterne (Normattiva/EUR-Lex), e va eseguito manualmente/offline
per popolare Qdrant. A runtime (api/routes.py) non viene mai contattata
alcuna fonte esterna, vedi CLAUDE.md.

--dry-run esegue lo scraping e stampa i conteggi senza scrivere su Qdrant:
utile per validare l'output degli scraper prima di un'ingestione completa.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time

from corpus.meta import write_last_update
from corpus.models import ArticleChunk
from corpus.scrapers.eurlex import EurLexScraper
from corpus.scrapers.normattiva import CODICI, NormattivaScraper
from search.embedder import Embedder
from search.retriever import build_retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_ingest(
    source: str,
    codici: list[str] | None,
    limit: int | None,
    dry_run: bool,
) -> int:
    """Esegue scraping + embedding + upsert su Qdrant.

    Ogni fonte (codice Normattiva, atto EUR-Lex) viene scaricata, incorporata
    e scritta su Qdrant IMMEDIATAMENTE, invece di accumulare tutti i chunk in
    memoria fino alla fine: su corpora lunghi (ore di scraping) un'interruzione
    o un errore di rete non fa perdere il lavoro già svolto."""
    start = time.monotonic()

    embedder = Embedder() if not dry_run else None
    retriever = build_retriever(embedder) if not dry_run else None
    if retriever is not None:
        retriever.ensure_collection()

    progress = {"n": 0}

    def on_progress() -> None:
        progress["n"] += 1
        if progress["n"] % 50 == 0:
            logger.info("Progresso: %d articoli processati finora", progress["n"])

    total_chunks = 0
    total_tokens = 0

    def flush(chunks: list[ArticleChunk], etichetta: str) -> None:
        nonlocal total_chunks, total_tokens
        if not chunks:
            return
        local_embedder = embedder or Embedder()
        total_tokens += local_embedder.count_tokens([c.testo for c in chunks])
        total_chunks += len(chunks)
        if retriever is not None:
            retriever.upsert_chunks(chunks)
            logger.info("Salvati su Qdrant %d chunk da %s", len(chunks), etichetta)
        else:
            logger.info("Dry-run: %d chunk da %s non scritti su Qdrant", len(chunks), etichetta)

    if source in ("normattiva", "all"):
        chiavi = codici or list(CODICI.keys())
        with NormattivaScraper() as scraper:
            for chiave in chiavi:
                chunks = scraper.fetch_codice(chiave, max_articoli=limit, on_progress=on_progress)
                flush(chunks, CODICI[chiave].nome)

    if source in ("eurlex", "all"):
        with EurLexScraper() as scraper:
            for tipo in ("regolamento", "direttiva"):
                for celex in scraper.discover_celex(tipo, limit=limit or 10):
                    chunks = scraper.fetch_act(celex, on_progress=on_progress)
                    flush(chunks, f"EUR-Lex {celex}")

    if total_chunks == 0:
        logger.error("Nessun chunk prodotto dagli scraper.")
        return 0

    elapsed = time.monotonic() - start
    logger.info("Documenti processati: %d", total_chunks)
    logger.info("Token totali: %d", total_tokens)
    logger.info("Tempo impiegato: %.1fs", elapsed)

    if not dry_run:
        write_last_update(total_chunks)

    return total_chunks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=["normattiva", "eurlex", "all"], required=True)
    parser.add_argument(
        "--codici",
        help="Sottoinsieme di codici Normattiva da scaricare, separati da virgola "
        f"(default: tutti). Disponibili: {', '.join(CODICI.keys())}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Normattiva: tetto massimo di articoli per codice. "
        "EUR-Lex: numero massimo di atti per tipo (regolamento/direttiva).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Non scrive su Qdrant.")
    args = parser.parse_args()

    codici = args.codici.split(",") if args.codici else None

    total = run_ingest(args.source, codici, args.limit, args.dry_run)
    if total == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
