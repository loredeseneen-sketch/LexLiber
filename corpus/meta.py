"""Metadata di ingestione del corpus (data/corpus_meta.json).

Qdrant non conserva un timestamp di "ultimo aggiornamento" della
collezione, ma `GET /health` deve mostrarlo. Invece di aggiungere
infrastruttura, usiamo un piccolo file JSON locale scritto da
`corpus/ingest.py` al termine di ogni ingestione riuscita.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

META_PATH = Path("data/corpus_meta.json")


def write_last_update(chunk_count: int) -> None:
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    META_PATH.write_text(
        json.dumps(
            {
                "ultimo_aggiornamento": datetime.now(timezone.utc).isoformat(),
                "chunk_processati_ultimo_run": chunk_count,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def read_last_update() -> str | None:
    """Restituisce il timestamp ISO dell'ultima ingestione riuscita, o None
    se il file non esiste ancora (es. corpus indicizzato prima
    dell'introduzione di questo tracking) o non è leggibile."""
    if not META_PATH.exists():
        return None
    try:
        return json.loads(META_PATH.read_text()).get("ultimo_aggiornamento")
    except Exception:
        return None
