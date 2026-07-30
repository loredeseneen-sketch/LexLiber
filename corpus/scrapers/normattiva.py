"""Scraper per i codici italiani pubblicati su Normattiva (normattiva.it),
la banca dati gratuita della normativa statale gestita dall'Istituto
Poligrafico e Zecca dello Stato.

Meccanismo reale (verificato contro il sito live): Normattiva non offre
una API pubblica. La pagina di un codice ("albero") mostra l'indice
completo degli articoli come una sequenza piatta di <li>, dove:

- gli header di raggruppamento (LIBRO/TITOLO/CAPO/Sezione) sono
  `<div class="collapse-header"><a ...>LIBRO QUARTO</a></div>
   <span class="snippets"><div ...>DELLE OBBLIGAZIONI</div></span>`;
- ogni articolo è un link
  `<a onclick="return showArticle('/atto/caricaArticolo?...', this);"
     class="numero_articolo">art. 1372</a>`
  la cui query string (idArticolo/idGruppo/codiceRedazionale/ecc.) è
  già completa e va usata così com'è per scaricare il testo via GET su
  `/atto/caricaArticolo?...` (endpoint AJAX che richiede solo la
  sessione stabilita caricando prima la pagina dell'albero — nessun
  parametro aggiuntivo o Referer necessario).

Il testo dell'articolo è dentro `span.attachment-just-text`, con un
primo `<div>` centrato contenente "Art. N. (Rubrica)." seguito dal corpo.

Se il markup cambia, aggiornare i pattern regex/selettori qui sotto.
Eseguito solo offline via `corpus/ingest.py`.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from bs4 import BeautifulSoup

from corpus.models import ArticleChunk
from corpus.scrapers.base import BaseScraper

logger = logging.getLogger(__name__)

BASE_URL = "https://www.normattiva.it"
# Osservato: la sessione stabilita da _load_tree scade dopo ~2h30 di uso
# continuo, causando una serie di HTTP 500 su /atto/caricaArticolo.
# Rinfrescarla ogni 500 articoli evita il problema su corpora lunghi.
SESSION_REFRESH_EVERY = 500


@dataclass(frozen=True)
class CodiceDef:
    chiave: str
    nome: str
    urn: str


CODICI: dict[str, CodiceDef] = {
    "codice_civile": CodiceDef(
        "codice_civile", "Codice Civile", "urn:nir:stato:regio.decreto:1942-03-16;262"
    ),
    "codice_procedura_civile": CodiceDef(
        "codice_procedura_civile",
        "Codice di Procedura Civile",
        "urn:nir:stato:regio.decreto:1940-10-28;1443",
    ),
    "codice_penale": CodiceDef(
        "codice_penale", "Codice Penale", "urn:nir:stato:regio.decreto:1930-10-19;1398"
    ),
    "codice_procedura_penale": CodiceDef(
        "codice_procedura_penale",
        "Codice di Procedura Penale",
        "urn:nir:stato:decreto.presidente.repubblica:1988-09-22;447",
    ),
    "dlgs_231_2001": CodiceDef(
        "dlgs_231_2001", "D.Lgs. 231/2001", "urn:nir:stato:decreto.legislativo:2001-06-08;231"
    ),
}

# Header di raggruppamento (LIBRO/PARTE, TITOLO, CAPO) nell'albero.
_HEADER_RE = re.compile(
    r'<a class="" data-toggle="collapse"[^>]*>([^<]*)</a></div>'
    r'<span class="snippets"><div id="[^"]*" class="collapse[^"]*">([^<]*)'
)
# Link ad un articolo, con la query string completa già pronta per
# /atto/caricaArticolo. L'etichetta può essere "art. N" oppure il solo
# numero N a seconda del codice; alcuni link con la stessa classe CSS
# sono in realtà voci di cronologia aggiornamenti ("agg.1", "orig.") e
# vanno scartate in _parse_tree.
_ARTICLE_LINK_RE = re.compile(
    r'<a href="" onclick="return showArticle\(\'([^\']+)\', this\);" '
    r'class="numero_articolo">\s*([^<]+?)\s*</a>'
)
# Etichette non-articolo da escludere (cronologia aggiornamenti/versione originale).
_NON_ARTICLE_LABEL_RE = re.compile(r"^(orig\.|agg\.\d+)$", re.IGNORECASE)
# Prefisso "art." opzionale da rimuovere per normalizzare il numero.
_ART_PREFIX_RE = re.compile(r"^art\.\s*", re.IGNORECASE)


class NormattivaScraper(BaseScraper):
    source_name = "normattiva"

    def fetch_codice(
        self,
        chiave: str,
        max_articoli: int | None = None,
        on_progress=None,
    ) -> list[ArticleChunk]:
        """Scarica gli articoli di un codice, nell'ordine dell'indice.

        `max_articoli`, se indicato, limita il numero di articoli scaricati
        (i primi N nell'ordine dell'indice), utile per test rapidi.
        `on_progress`, se indicato, viene chiamato dopo ogni tentativo di
        scaricare un articolo (riuscito o no) — usato da ingest.py per
        loggare l'avanzamento su corpora lunghi.
        """
        if chiave not in CODICI:
            raise ValueError(f"Codice sconosciuto: {chiave!r}. Disponibili: {list(CODICI)}")
        definizione = CODICI[chiave]

        tree_html = self._load_tree(definizione)
        voci = self._parse_tree(tree_html)
        if not voci:
            logger.error(
                "Nessun articolo trovato nell'indice di %s: il markup di Normattiva "
                "potrebbe essere cambiato (vedi _HEADER_RE/_ARTICLE_LINK_RE).",
                definizione.nome,
            )
            return []

        if max_articoli is not None:
            voci = voci[:max_articoli]

        chunks: list[ArticleChunk] = []
        for i, voce in enumerate(voci):
            # Rinfresca la sessione proattivamente ogni SESSION_REFRESH_EVERY
            # articoli: su corpora lunghi (migliaia di articoli, ore di
            # scraping) la sessione stabilita da _load_tree scade, causando
            # altrimenti una lunga serie di HTTP 500 consecutivi.
            if i > 0 and i % SESSION_REFRESH_EVERY == 0:
                logger.info(
                    "Rinfresco proattivo della sessione dopo %d articoli (%s)",
                    i,
                    definizione.nome,
                )
                self._load_tree(definizione)

            chunk = self._fetch_article(definizione, voce)
            if chunk is not None:
                chunks.append(chunk)
            if on_progress is not None:
                on_progress()

        logger.info(
            "Normattiva: %d/%d articoli estratti da %s",
            len(chunks),
            len(voci),
            definizione.nome,
        )
        return chunks

    def _load_tree(self, definizione: CodiceDef) -> str:
        """Carica la pagina indice del codice. Questa richiesta stabilisce
        anche la sessione (cookie) necessaria alle successive chiamate a
        /atto/caricaArticolo sullo stesso client."""
        url = f"{BASE_URL}/uri-res/N2Ls?{definizione.urn}"
        resp = self._get(url)
        return resp.text

    def _parse_tree(self, html: str) -> list[dict]:
        """Scansiona l'albero in ordine di documento, tenendo traccia
        dell'ultimo LIBRO/TITOLO/CAPO incontrato e assegnandolo a ogni
        articolo che segue."""
        eventi = [(m.start(), "header", m) for m in _HEADER_RE.finditer(html)]
        eventi += [(m.start(), "articolo", m) for m in _ARTICLE_LINK_RE.finditer(html)]
        eventi.sort(key=lambda e: e[0])

        libro = titolo = capo = None
        voci: list[dict] = []
        for _, tipo, m in eventi:
            if tipo == "header":
                etichetta, descrizione = m.group(1).strip(), m.group(2).strip()
                testo_header = f"{etichetta} - {descrizione}" if descrizione else etichetta
                etichetta_upper = etichetta.upper()
                if etichetta_upper.startswith("LIBRO") or etichetta_upper.startswith("PARTE"):
                    libro, titolo, capo = testo_header, None, None
                elif etichetta_upper.startswith("TITOLO"):
                    titolo, capo = testo_header, None
                elif etichetta_upper.startswith("CAPO"):
                    capo = testo_header
            else:
                query_path, numero_raw = m.group(1), m.group(2).strip()
                if _NON_ARTICLE_LABEL_RE.match(numero_raw):
                    continue
                numero = _ART_PREFIX_RE.sub("", numero_raw).strip()
                voci.append(
                    {
                        "numero": numero,
                        "query_path": query_path,
                        "libro": libro,
                        "titolo": titolo,
                        "capo": capo,
                    }
                )
        return voci

    def _extract_plain_text(self, soup: BeautifulSoup) -> tuple[str, str] | None:
        """Formato "classico" (es. Codice Civile/Penale): il testo intero,
        header compreso, sta in `span.attachment-just-text`."""
        container = soup.select_one("span.attachment-just-text")
        if container is None:
            return None
        header_div = container.find("div")
        articolo_label = header_div.get_text(" ", strip=True) if header_div else ""
        if header_div:
            header_div.extract()
        testo = container.get_text("\n", strip=True)
        return articolo_label, testo

    def _extract_akn_text(self, soup: BeautifulSoup) -> tuple[str, str] | None:
        """Formato "Akoma Ntoso" (es. Codice di Procedura Penale, D.Lgs.
        231/2001): numero/rubrica ed i singoli commi sono marcati con
        classi `*-akn` separate, senza un unico blocco di testo piatto."""
        numero_tag = soup.select_one("h2.article-num-akn")
        if numero_tag is None:
            return None
        numero_testo = numero_tag.get_text(" ", strip=True)
        heading_tag = soup.select_one("div.article-heading-akn")
        heading_testo = heading_tag.get_text(" ", strip=True) if heading_tag else ""
        articolo_label = f"{numero_testo}. ({heading_testo})." if heading_testo else numero_testo

        commi = soup.select("div.art-comma-div-akn")
        righe = [c.get_text(" ", strip=True) for c in commi]
        righe = [r for r in righe if r]
        testo = "\n".join(righe)
        return articolo_label, testo

    def _fetch_article(self, definizione: CodiceDef, voce: dict) -> ArticleChunk | None:
        url = f"{BASE_URL}{voce['query_path']}"
        try:
            resp = self._get(url)
        except Exception:
            # La sessione (cookie) stabilita da _load_tree scade dopo un uso
            # prolungato (osservato: ~2h30 di scraping continuo, poi ogni
            # richiesta a /atto/caricaArticolo torna HTTP 500). Un solo
            # retry con sessione rinfrescata basta a recuperare.
            logger.info(
                "Articolo %s: richiesta fallita per %s, rinfresco la sessione e riprovo",
                voce["numero"],
                definizione.nome,
            )
            self._load_tree(definizione)
            try:
                resp = self._get(url)
            except Exception:
                logger.warning(
                    "Articolo %s non raggiungibile per %s (anche dopo refresh sessione)",
                    voce["numero"],
                    definizione.nome,
                )
                return None

        soup = BeautifulSoup(resp.text, "lxml")
        estratto = self._extract_plain_text(soup) or self._extract_akn_text(soup)
        if estratto is None:
            logger.debug(
                "Articolo %s privo di testo per %s (probabilmente abrogato)",
                voce["numero"],
                definizione.nome,
            )
            return None
        articolo_label, testo = estratto
        if not testo:
            return None

        permalink_suffix = re.sub(r"\s+", "", voce["numero"])
        url_originale = f"{BASE_URL}/uri-res/N2Ls?{definizione.urn}~art{permalink_suffix}!vig="

        return ArticleChunk(
            fonte=self.source_name,
            codice=definizione.nome,
            libro=voce["libro"],
            titolo=voce["titolo"],
            capo=voce["capo"],
            articolo=articolo_label,
            numero_articolo=voce["numero"],
            testo=testo,
            url_originale=url_originale,
        )

    def search(self, query: str, max_results: int = 20) -> list[ArticleChunk]:
        raise NotImplementedError(
            "NormattivaScraper.search non è implementato in questa sessione: "
            "i codici si scaricano per intero con fetch_codice(), non per parola chiave."
        )
