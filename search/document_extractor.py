"""Estrazione testo locale da documenti caricati dall'utente (PDF/DOCX).

Usato dalla sezione "Analizza il mio documento" (vedi chat/engine.py:
ChatEngine.analyze_document). L'estrazione avviene interamente in locale
con pypdf/python-docx: nessuna chiamata esterna, coerente con SOVEREIGN
(CLAUDE.md) — il file caricato non lascia mai il server.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

from docx import Document as DocxDocument
from pypdf import PdfReader

from config import settings

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


class UnsupportedFileTypeError(ValueError):
    """Sollevata per estensioni diverse da .pdf/.docx."""


class EmptyDocumentError(ValueError):
    """Sollevata quando l'estrazione non produce testo utilizzabile
    (es. PDF scansionato senza livello testuale/OCR)."""


def extract_text(filename: str, content: bytes) -> str:
    """Estrae il testo da un PDF o DOCX, troncandolo a
    `settings.document_max_chars` se necessario.

    Il dispatch avviene sull'estensione del nome file (non sul content-type
    HTTP, spesso inattendibile per upload da browser diversi)."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        text = _extract_pdf(content)
    elif suffix == ".docx":
        text = _extract_docx(content)
    else:
        raise UnsupportedFileTypeError(
            f"Estensione {suffix!r} non supportata. Formati accettati: PDF, DOCX."
        )

    text = text.strip()
    if not text:
        raise EmptyDocumentError(
            "Nessun testo estraibile dal documento (es. PDF scansionato senza OCR)."
        )

    max_chars = settings.document_max_chars
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[...documento troncato: superava il limite di analisi...]"
    return text


def _extract_pdf(content: bytes) -> str:
    reader = PdfReader(BytesIO(content))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(p for p in pages if p.strip())


def _extract_docx(content: bytes) -> str:
    doc = DocxDocument(BytesIO(content))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)
