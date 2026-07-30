"""Scraper per le massime pubbliche della Corte di Cassazione, tramite il
portale "Italgiure" pubblico (cortedicassazione.it / italgiure.giustizia.it
- sezione massimario pubblico, non l'area riservata agli avvocati).

Espone solo le sentenze e massime rese pubbliche gratuitamente dalla Corte.
Eseguito solo offline via `corpus/ingest.py`.
"""
from __future__ import annotations

import logging
from urllib.parse import quote

from bs4 import BeautifulSoup

from corpus.models import LegalDocument
from corpus.scrapers.base import BaseScraper

logger = logging.getLogger(__name__)

BASE_URL = "https://www.cortedicassazione.it"
SEARCH_URL = BASE_URL + "/corte-di-cassazione/it/ricerca.page?query={query}"


class CassazioneScraper(BaseScraper):
    source_name = "cassazione"

    def search(self, query: str, max_results: int = 20) -> list[LegalDocument]:
        url = SEARCH_URL.format(query=quote(query))
        try:
            resp = self._get(url)
        except Exception:
            logger.exception("Ricerca Cassazione fallita per query=%r", query)
            return []

        soup = BeautifulSoup(resp.text, "lxml")
        results: list[LegalDocument] = []

        for row in soup.select("div.risultato-ricerca, li.search-result")[:max_results]:
            link = row.select_one("a[href]")
            if not link:
                continue
            title = link.get_text(strip=True)
            href = link["href"]
            doc_url = href if href.startswith("http") else BASE_URL + href

            excerpt_el = row.select_one("p, div.massima")
            text = excerpt_el.get_text(strip=True) if excerpt_el else ""

            date_el = row.select_one("span.data, time")
            date = date_el.get_text(strip=True) if date_el else None

            results.append(
                LegalDocument(
                    source=self.source_name,
                    title=title or query,
                    identifier=title or query,
                    url=doc_url,
                    text=text,
                    date=date,
                    metadata={"query": query},
                )
            )

        logger.info("Cassazione: %d risultati per %r", len(results), query)
        return results
