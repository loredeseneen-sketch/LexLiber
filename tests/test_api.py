"""Test dell'endpoint /api/ask. Il ChatEngine viene sostituito con un
doppio di test per non dipendere da Qdrant/Claude API durante i test.
"""
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import api.routes as routes_module
from chat.engine import ChatAnswer
from search.citation import Citation


@pytest.fixture
def client(monkeypatch):
    fake_engine = MagicMock()
    monkeypatch.setattr(routes_module, "_engine", fake_engine)
    monkeypatch.setattr(routes_module, "get_engine", lambda: fake_engine)
    return TestClient(routes_module.app), fake_engine


def test_health(client):
    test_client, _ = client
    resp = test_client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ask_returns_citations_when_sufficient(client):
    test_client, fake_engine = client
    fake_engine.ask.return_value = ChatAnswer(
        answer="Il recesso è disciplinato dall'art. 1373 c.c.",
        citations=[
            Citation(
                source="normattiva",
                identifier="Art. 1373 c.c.",
                url="https://www.normattiva.it/x",
                excerpt="Se a una delle parti è attribuita la facoltà di recesso...",
                score=0.91,
            )
        ],
        sufficient_sources=True,
    )

    resp = test_client.post("/api/ask", json={"question": "Come funziona il recesso?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["sufficient_sources"] is True
    assert len(body["citations"]) == 1
    assert body["citations"][0]["url"] == "https://www.normattiva.it/x"


def test_ask_returns_fallback_when_insufficient(client):
    test_client, fake_engine = client
    fake_engine.ask.return_value = ChatAnswer(
        answer="Non ho trovato fonti sufficienti per rispondere con certezza.",
        citations=[],
        sufficient_sources=False,
    )

    resp = test_client.post("/api/ask", json={"question": "Domanda oscura senza fonti"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["sufficient_sources"] is False
    assert body["citations"] == []


def test_ask_rejects_too_short_question(client):
    test_client, _ = client
    resp = test_client.post("/api/ask", json={"question": "ok"})
    assert resp.status_code == 422


def test_ask_returns_500_on_internal_error(client):
    test_client, fake_engine = client
    fake_engine.ask.side_effect = RuntimeError("boom")

    resp = test_client.post("/api/ask", json={"question": "Domanda valida qui"})
    assert resp.status_code == 500


def test_search_returns_fonti_and_confidence_when_sufficient(client):
    test_client, fake_engine = client
    fake_engine.ask.return_value = ChatAnswer(
        answer="Il recesso è disciplinato dall'art. 1373 c.c.",
        citations=[
            Citation(
                source="normattiva",
                identifier="Art. 1373 c.c.",
                url="https://www.normattiva.it/x",
                excerpt="Se a una delle parti è attribuita la facoltà di recesso...",
                score=0.91,
                codice="Codice Civile",
                articolo="Art. 1373",
            )
        ],
        sufficient_sources=True,
    )

    resp = test_client.post("/search", json={"query": "Come funziona il recesso?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["confidence"] == "alta"
    assert body["risposta"] == "Il recesso è disciplinato dall'art. 1373 c.c."
    assert len(body["fonti"]) == 1
    assert body["fonti"][0]["url"] == "https://www.normattiva.it/x"
    assert body["fonti"][0]["label"] == "Art. 1373, Codice Civile"
    assert isinstance(body["tempo_ms"], int)


def test_search_returns_bassa_confidence_when_insufficient(client):
    test_client, fake_engine = client
    fake_engine.ask.return_value = ChatAnswer(
        answer="Non ho trovato fonti sufficienti per rispondere con certezza.",
        citations=[],
        sufficient_sources=False,
    )

    resp = test_client.post("/search", json={"query": "Domanda oscura senza fonti"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["confidence"] == "bassa"
    assert body["fonti"] == []


def test_analyze_document_returns_risposta_and_fonti_normative(client, monkeypatch):
    test_client, fake_engine = client
    fake_engine.analyze_document.return_value = ChatAnswer(
        answer="Il documento contiene una clausola di recesso [Documento caricato], "
        "disciplinata dall'art. 1373 c.c. [Art. 1373, Codice Civile]",
        citations=[
            Citation(
                source="normattiva",
                identifier="Art. 1373 c.c.",
                url="https://www.normattiva.it/x",
                excerpt="Se a una delle parti è attribuita la facoltà di recesso...",
                score=0.91,
                codice="Codice Civile",
                articolo="Art. 1373",
            )
        ],
        sufficient_sources=True,
    )

    from io import BytesIO

    from docx import Document as DocxDocument

    doc = DocxDocument()
    doc.add_paragraph("Contratto di locazione con clausola di recesso.")
    buf = BytesIO()
    doc.save(buf)
    buf.seek(0)

    resp = test_client.post(
        "/analyze-document",
        data={"question": "Come funziona il recesso in questo contratto?"},
        files={"file": ("contratto.docx", buf.read(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["confidence"] == "alta"
    assert body["documento"] == "contratto.docx"
    assert len(body["fonti_normative"]) == 1
    assert body["fonti_normative"][0]["url"] == "https://www.normattiva.it/x"
    fake_engine.analyze_document.assert_called_once()


def test_analyze_document_rejects_unsupported_extension(client):
    test_client, _ = client
    resp = test_client.post(
        "/analyze-document",
        data={"question": "Domanda valida qui"},
        files={"file": ("note.txt", b"testo semplice", "text/plain")},
    )
    assert resp.status_code == 400


def test_analyze_document_rejects_too_short_question(client):
    test_client, _ = client
    resp = test_client.post(
        "/analyze-document",
        data={"question": "ok"},
        files={"file": ("note.txt", b"testo semplice", "text/plain")},
    )
    assert resp.status_code == 422


def test_health_full_reports_status_ok(client):
    test_client, fake_engine = client
    fake_engine.retriever.client.count.return_value.count = 6285

    resp = test_client.get("/health")

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["chunk_totali"] == 6285
    assert "ultimo_aggiornamento" in body
