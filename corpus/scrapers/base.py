"""Classe base per gli scraper del corpus.

IMPORTANTE (vedi CLAUDE.md, principio SOVEREIGN): questi scraper girano
esclusivamente offline/batch, lanciati manualmente tramite
`corpus/ingest.py`, per popolare Qdrant una tantum (o periodicamente).
A runtime, quando un utente interroga il sistema, non viene mai fatta
alcuna chiamata di rete verso Normattiva/EUR-Lex/Cassazione: si interroga
solo l'indice Qdrant locale già popolato.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod

import httpx

from corpus.models import LegalDocument

logger = logging.getLogger(__name__)

USER_AGENT = "LexLiber/1.0 (self-hosted legal research tool; offline batch ingest)"


class BaseScraper(ABC):
    """Interfaccia comune: ogni scraper implementa `search()`.

    Applica un rate-limit conservativo tra le richieste per non gravare
    sui server pubblici delle fonti istituzionali.
    """

    source_name: str = "base"
    min_delay_seconds: float = 1.5

    def __init__(self, timeout: float = 20.0):
        self._client = httpx.Client(
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
            follow_redirects=True,
        )
        self._last_request_ts: float = 0.0

    def _get(self, url: str, **kwargs) -> httpx.Response:
        self._throttle()
        logger.info("GET %s", url)
        resp = self._client.get(url, **kwargs)
        resp.raise_for_status()
        return resp

    def _post(self, url: str, **kwargs) -> httpx.Response:
        self._throttle()
        logger.info("POST %s", url)
        resp = self._client.post(url, **kwargs)
        resp.raise_for_status()
        return resp

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_ts
        wait = self.min_delay_seconds - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_ts = time.monotonic()

    @abstractmethod
    def search(self, query: str, max_results: int = 20) -> list[LegalDocument]:
        """Cerca `query` sulla fonte e restituisce documenti normalizzati."""
        raise NotImplementedError

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
