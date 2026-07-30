"""Test del ChatEngine: verifica che il LLM non venga MAI chiamato quando
il citation gate non trova fonti sufficienti (regola CITATION, CLAUDE.md).
"""
from unittest.mock import MagicMock

from chat.engine import ChatEngine
from search.citation import NO_SOURCES_MESSAGE
from search.retriever import RetrievedPassage


def make_passage(score: float) -> RetrievedPassage:
    return RetrievedPassage(
        score=score,
        doc_id="doc-1",
        source="normattiva",
        title="Art. 1372 c.c.",
        identifier="Art. 1372 c.c.",
        url="https://www.normattiva.it/x",
        text="Il contratto ha forza di legge tra le parti.",
    )


def test_engine_returns_fallback_without_calling_llm(monkeypatch):
    retriever = MagicMock()
    retriever.search.return_value = [make_passage(0.3)]

    engine = ChatEngine(retriever=retriever)
    engine._generate_claude = MagicMock(side_effect=AssertionError("LLM non doveva essere chiamato"))
    engine._generate_ollama = MagicMock(side_effect=AssertionError("LLM non doveva essere chiamato"))

    result = engine.ask("Domanda qualsiasi")

    assert result.sufficient_sources is False
    assert result.answer == NO_SOURCES_MESSAGE
    assert result.citations == []
    engine._generate_claude.assert_not_called()
    engine._generate_ollama.assert_not_called()


def test_engine_calls_llm_and_returns_citations_when_sufficient(monkeypatch):
    # Il provider LLM effettivo è config-dipendente (vedi .env, LLM_PROVIDER);
    # qui mockiamo entrambi i generatori e forziamo esplicitamente il
    # provider così il test non dipende dall'ambiente in cui gira.
    from config import settings

    monkeypatch.setattr(settings, "llm_provider", "claude")

    retriever = MagicMock()
    retriever.search.return_value = [make_passage(0.9)]

    engine = ChatEngine(retriever=retriever)
    engine._generate_claude = MagicMock(return_value="Risposta sintetica basata sulla fonte.")
    engine._generate_ollama = MagicMock(side_effect=AssertionError("Ollama non doveva essere chiamato"))

    result = engine.ask("Cosa dice l'art. 1372 c.c.?")

    assert result.sufficient_sources is True
    assert result.answer == "Risposta sintetica basata sulla fonte."
    assert len(result.citations) == 1
    assert result.citations[0].url == "https://www.normattiva.it/x"
    engine._generate_claude.assert_called_once()
    engine._generate_ollama.assert_not_called()


def test_analyze_document_includes_citations_when_relevant_norms_found():
    from config import settings

    retriever = MagicMock()
    retriever.search.return_value = [make_passage(0.9)]

    engine = ChatEngine(retriever=retriever)
    engine._generate_claude = MagicMock(return_value="Il contratto richiama l'art. 1372 c.c.")
    engine._generate_ollama = MagicMock(side_effect=AssertionError("Ollama non doveva essere chiamato"))

    result = engine.analyze_document(
        "Questa clausola è vincolante?", "Testo del contratto caricato dall'utente."
    )

    assert result.sufficient_sources is True
    assert len(result.citations) == 1
    engine._generate_claude.assert_called_once()
    # Il prompt di sistema usato deve essere quello per l'analisi documento,
    # non quello generico di ricerca normativa.
    call_kwargs = engine._generate_claude.call_args
    assert "documento" in call_kwargs.args[0].lower() or "documento" in str(call_kwargs).lower()


def test_analyze_document_still_answers_without_matching_norms():
    """A differenza di ask(), l'assenza di norme rilevanti non deve bloccare
    la generazione: il documento caricato resta una fonte legittima."""
    retriever = MagicMock()
    retriever.search.return_value = [make_passage(0.1)]  # sotto soglia citation gate

    engine = ChatEngine(retriever=retriever)
    engine._generate_claude = MagicMock(return_value="Analisi basata solo sul documento caricato.")
    engine._generate_ollama = MagicMock(side_effect=AssertionError("Ollama non doveva essere chiamato"))

    result = engine.analyze_document("Di cosa parla questo documento?", "Testo del documento.")

    assert result.sufficient_sources is False
    assert result.citations == []
    assert result.answer == "Analisi basata solo sul documento caricato."
    engine._generate_claude.assert_called_once()
