"""Scraper per EUR-Lex (eur-lex.europa.eu), il portale ufficiale e gratuito
del diritto dell'Unione Europea.

Gli atti da scaricare (regolamenti in vigore, direttive) vengono
individuati tramite l'endpoint SPARQL pubblico di CELLAR (nessuna
registrazione/API key richiesta): https://publications.europa.eu/webapi/rdf/sparql
Il testo integrale in italiano viene poi scaricato dalla pagina HTML
pubblica di EUR-Lex e spezzato articolo per articolo con la stessa
struttura chunk usata per Normattiva.

Se l'endpoint SPARQL non è raggiungibile (rete ristretta dello studio,
cambio di schema CELLAR), si usa un elenco di riserva di CELEX noti e
pubblici (`FALLBACK_CELEX`), così la pipeline di ingest resta eseguibile
anche senza accesso a CELLAR. Eseguito solo offline via
`corpus/ingest.py`.
"""
from __future__ import annotations

import logging

from bs4 import BeautifulSoup

from corpus.models import ArticleChunk
from corpus.scrapers.base import BaseScraper
from corpus.scrapers.structure import extract_structure_labels, split_articles_from_text

logger = logging.getLogger(__name__)

SPARQL_ENDPOINT = "https://publications.europa.eu/webapi/rdf/sparql"
EURLEX_TEXT_URL = "https://eur-lex.europa.eu/legal-content/IT/TXT/HTML/?uri=CELEX:{celex}"

# Contenitori del testo principale della pagina EUR-Lex, in ordine di
# preferenza: usati per escludere menu/indice e ridurre il rischio che lo
# split per "Articolo N" trovi anche le voci del sommario.
_MAIN_CONTENT_SELECTORS = ("div#TexteOnly", "div.tabConsLegReport", "div#text", "body")

_TIPO_RESOURCE = {"regolamento": "REG", "direttiva": "DIR"}

# Regolamenti/direttive UE pubblici e in vigore, usati come riserva se la
# query SPARQL su CELLAR non è raggiungibile dalla rete dello studio.
FALLBACK_CELEX: dict[str, list[str]] = {
    "regolamento": [
        "32016R0679",  # GDPR
        "32022R2065",  # Digital Services Act
        "32022R1925",  # Digital Markets Act
        "32024R1689",  # AI Act
    ],
    "direttiva": [
        "32019L0771",  # vendita di beni di consumo
        "32015L2366",  # PSD2
    ],
}

_SPARQL_QUERY = """
PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
SELECT DISTINCT ?celex WHERE {{
  ?work cdm:resource_legal_id_celex ?celex .
  ?work cdm:work_has_resource-type <http://publications.europa.eu/resource/authority/resource-type/{tipo}> .
  ?work cdm:resource_legal_in-force "true"^^<http://www.w3.org/2001/XMLSchema#boolean> .
}}
LIMIT {limit}
"""


class EurLexScraper(BaseScraper):
    source_name = "eurlex"

    def discover_celex(self, tipo: str, limit: int = 20) -> list[str]:
        """Interroga CELLAR via SPARQL per ottenere i CELEX degli atti del
        tipo richiesto ("regolamento" | "direttiva"). Ricade sull'elenco
        di riserva se la query fallisce o non produce risultati."""
        query = _SPARQL_QUERY.format(tipo=_TIPO_RESOURCE[tipo], limit=limit)
        try:
            resp = self._post(
                SPARQL_ENDPOINT,
                data={"query": query, "format": "json"},
                headers={"Accept": "application/sparql-results+json"},
            )
            bindings = resp.json()["results"]["bindings"]
            celex_ids = [b["celex"]["value"] for b in bindings]
            if celex_ids:
                return celex_ids
            logger.warning("Query SPARQL su CELLAR senza risultati per tipo=%s", tipo)
        except Exception:
            logger.warning(
                "Query SPARQL su CELLAR non riuscita per tipo=%s, uso l'elenco di riserva", tipo
            )
        return FALLBACK_CELEX.get(tipo, [])

    def fetch_act(self, celex: str, on_progress=None) -> list[ArticleChunk]:
        """Scarica il testo integrale in italiano dell'atto CELEX indicato
        e lo spezza in un ArticleChunk per articolo.

        `on_progress`, se indicato, viene chiamato una volta per ogni
        articolo estratto — usato da ingest.py per loggare l'avanzamento."""
        url = EURLEX_TEXT_URL.format(celex=celex)
        try:
            resp = self._get(url)
        except Exception:
            logger.warning("Impossibile scaricare CELEX %s", celex)
            return []

        soup = BeautifulSoup(resp.text, "lxml")
        titolo_atto = self._extract_title(soup) or celex
        full_text = self._extract_main_text(soup)

        raw_articles = split_articles_from_text(full_text)
        # In presenza di un indice/sommario prima del testo integrale può
        # comparire più di una voce per lo stesso numero_articolo: si tiene
        # quella con il testo più lungo, tipicamente il corpo reale.
        by_numero: dict[str, tuple[str, str]] = {}
        for numero, header, testo in raw_articles:
            existing = by_numero.get(numero)
            if existing is None or len(testo) > len(existing[1]):
                by_numero[numero] = (header, testo)

        chunks: list[ArticleChunk] = []
        for numero, (header, testo) in by_numero.items():
            preceding = full_text[: full_text.rfind(header)]
            struttura = extract_structure_labels(preceding)
            chunks.append(
                ArticleChunk(
                    fonte=self.source_name,
                    codice=titolo_atto,
                    libro=struttura["libro"],
                    titolo=struttura["titolo"],
                    capo=struttura["capo"],
                    articolo=header,
                    numero_articolo=numero,
                    testo=testo,
                    url_originale=url,
                )
            )
            if on_progress is not None:
                on_progress()

        logger.info("EUR-Lex: %d articoli estratti da CELEX %s", len(chunks), celex)
        return chunks

    def fetch_all(
        self,
        tipi: list[str] | None = None,
        limit_per_tipo: int = 10,
        on_progress=None,
    ) -> list[ArticleChunk]:
        """Scopre e scarica tutti gli atti dei tipi richiesti."""
        tipi = tipi or list(_TIPO_RESOURCE.keys())
        chunks: list[ArticleChunk] = []
        for tipo in tipi:
            for celex in self.discover_celex(tipo, limit=limit_per_tipo):
                chunks.extend(self.fetch_act(celex, on_progress=on_progress))
        return chunks

    def _extract_main_text(self, soup: BeautifulSoup) -> str:
        for selector in _MAIN_CONTENT_SELECTORS:
            el = soup.select_one(selector)
            if el:
                return el.get_text("\n")
        return soup.get_text("\n")

    def _extract_title(self, soup: BeautifulSoup) -> str | None:
        main_title = soup.select_one("div.eli-main-title")
        if main_title:
            righe = [p.get_text(" ", strip=True) for p in main_title.select("p.oj-doc-ti")]
            righe = [r for r in righe if r]
            if righe:
                return " ".join(righe)
        el = soup.select_one("p.doc-ti") or soup.title
        return el.get_text(" ", strip=True) if el else None

    def search(self, query: str, max_results: int = 20) -> list[ArticleChunk]:
        raise NotImplementedError(
            "EurLexScraper.search non è implementato in questa sessione: "
            "usare fetch_all()/fetch_act()."
        )
