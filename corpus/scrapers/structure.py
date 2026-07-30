"""Helper di parsing condivisi tra gli scraper Normattiva ed EUR-Lex.

Sono funzioni pure (nessuna chiamata di rete), quindi facilmente testabili
in isolamento — vedi tests/test_scraper_structure.py.
"""
from __future__ import annotations

import re

# Cerca l'ultima intestazione di ciascun livello (LIBRO/PARTE, TITOLO, CAPO)
# in un blocco di testo. Robusto a variazioni di markup perché lavora sul
# testo visibile della pagina, non su selettori CSS fragili.
_ORDINALI = r"PRIMO|SECONDO|TERZO|QUARTO|QUINTO|SESTO|SETTIMO|OTTAVO|NONO|DECIMO"

_STRUCTURE_PATTERNS = {
    # LIBRO nel Codice Civile usa ordinali per esteso (es. "LIBRO QUARTO"),
    # non numeri romani; PARTE (es. nel c.p.p.) usa ordinali femminili.
    "libro": re.compile(
        rf"(?:LIBRO\s+(?:{_ORDINALI}|[IVXLCDM]+)|PARTE\s+(?:PRIMA|SECONDA|TERZA|QUARTA|QUINTA))\b[^\n]*",
        re.IGNORECASE,
    ),
    "titolo": re.compile(r"TITOLO\s+[IVXLCDM]+\b[^\n]*", re.IGNORECASE),
    "capo": re.compile(r"CAPO\s+[IVXLCDM]+\b[^\n]*", re.IGNORECASE),
}


def extract_structure_labels(text: str) -> dict[str, str | None]:
    """Ricava libro/titolo/capo dall'ultima occorrenza di ciascun header
    trovata in `text` (tipicamente il testo che precede l'articolo).

    Ritorna sempre le tre chiavi "libro", "titolo", "capo" (None se non
    trovate) così che il chiamante possa costruire un ArticleChunk senza
    controlli aggiuntivi.
    """
    result: dict[str, str | None] = {}
    for key, pattern in _STRUCTURE_PATTERNS.items():
        matches = pattern.findall(text)
        result[key] = matches[-1].strip() if matches else None
    return result


# Individua intestazioni "Articolo N" (o "Articolo Nbis") su una riga
# propria, usato per spezzare il testo integrale di un atto UE in articoli.
_ARTICLE_HEADER_RE = re.compile(r"^[ \t]*Articolo\s+(\d+[a-zA-Z]?)[ \t]*$", re.IGNORECASE | re.MULTILINE)


def split_articles_from_text(full_text: str) -> list[tuple[str, str, str]]:
    """Spezza il testo integrale di un atto nei suoi articoli.

    Ritorna una lista di tuple (numero_articolo, header_completo, testo).
    Nota: se il testo di origine contiene un indice/sommario con gli stessi
    header "Articolo N" prima del testo vero e proprio (comune nelle pagine
    EUR-Lex), può produrre voci duplicate con testo vuoto/breve: il
    chiamante dovrebbe scartare le voci con testo troppo corto o duplicati
    per numero_articolo tenendo l'ultima occorrenza.
    """
    matches = list(_ARTICLE_HEADER_RE.finditer(full_text))
    results: list[tuple[str, str, str]] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        numero = m.group(1)
        header = m.group(0).strip()
        testo = full_text[start:end].strip()
        if testo:
            results.append((numero, header, testo))
    return results
