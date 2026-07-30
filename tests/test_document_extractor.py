"""Test dell'estrazione locale di testo da PDF/DOCX (search/document_extractor.py)."""
from io import BytesIO

import pytest
from docx import Document as DocxDocument
from pypdf import PdfWriter

from search.document_extractor import (
    EmptyDocumentError,
    UnsupportedFileTypeError,
    extract_text,
)


def _make_docx_bytes(paragraphs: list[str]) -> bytes:
    doc = DocxDocument()
    for p in paragraphs:
        doc.add_paragraph(p)
    buf = BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _make_empty_pdf_bytes() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_extract_text_docx_returns_paragraphs():
    content = _make_docx_bytes(["Contratto di locazione", "Articolo 1: oggetto del contratto"])
    text = extract_text("contratto.docx", content)
    assert "Contratto di locazione" in text
    assert "Articolo 1" in text


def test_extract_text_rejects_unsupported_extension():
    with pytest.raises(UnsupportedFileTypeError):
        extract_text("note.txt", b"qualche testo")


def test_extract_text_raises_on_empty_pdf():
    with pytest.raises(EmptyDocumentError):
        extract_text("scansione.pdf", _make_empty_pdf_bytes())


def test_extract_text_truncates_long_document(monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "document_max_chars", 50)
    content = _make_docx_bytes(["a" * 200])
    text = extract_text("lungo.docx", content)
    assert len(text) <= 50 + len("\n\n[...documento troncato: superava il limite di analisi...]")
    assert "troncato" in text
