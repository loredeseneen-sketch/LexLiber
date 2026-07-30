"""Test del citation gate — il componente più critico del sistema
(vedi CLAUDE.md, regola CITATION): zero output senza fonte citata.
"""
from search.citation import NO_SOURCES_MESSAGE, apply_citation_gate, build_excerpt, format_citation
from search.retriever import RetrievedPassage


def make_passage(score: float, text: str = "Testo di esempio.\nSeconda riga.\nTerza riga.\nQuarta riga.") -> RetrievedPassage:
    return RetrievedPassage(
        score=score,
        doc_id="doc-1",
        source="normattiva",
        title="Codice Civile, art. 1372",
        identifier="Art. 1372 c.c.",
        url="https://www.normattiva.it/esempio",
        text=text,
        codice="Codice Civile",
        articolo="Art. 1372",
    )


def test_gate_blocks_when_no_passage_above_threshold():
    passages = [make_passage(0.5), make_passage(0.6)]
    result = apply_citation_gate(passages, min_score=0.75)

    assert result.sufficient is False
    assert result.citations == []
    assert result.fallback_message == NO_SOURCES_MESSAGE


def test_gate_blocks_on_empty_passages():
    result = apply_citation_gate([], min_score=0.75)
    assert result.sufficient is False
    assert result.fallback_message == NO_SOURCES_MESSAGE


def test_gate_allows_when_passage_meets_threshold():
    passages = [make_passage(0.9), make_passage(0.5)]
    result = apply_citation_gate(passages, min_score=0.75)

    assert result.sufficient is True
    assert len(result.citations) == 1
    assert result.citations[0].source == "normattiva"
    assert result.citations[0].url == "https://www.normattiva.it/esempio"


def test_gate_is_exact_at_threshold_boundary():
    result = apply_citation_gate([make_passage(0.75)], min_score=0.75)
    assert result.sufficient is True


def test_excerpt_is_truncated_to_max_lines():
    text = "Riga 1\nRiga 2\nRiga 3\nRiga 4\nRiga 5"
    excerpt = build_excerpt(text, max_lines=3)
    assert excerpt.count("\n") == 2
    assert "Riga 4" not in excerpt


def test_citation_to_dict_has_required_fields():
    result = apply_citation_gate([make_passage(0.9)], min_score=0.75)
    payload = result.citations[0].to_dict()
    assert set(payload.keys()) == {"source", "identifier", "url", "excerpt", "score"}


def test_citation_label_uses_articolo_and_codice():
    result = apply_citation_gate([make_passage(0.9)], min_score=0.75)
    assert result.citations[0].label == "Art. 1372, Codice Civile"


def test_citation_label_falls_back_to_identifier_without_codice():
    from search.citation import Citation

    citation = Citation(
        source="normattiva",
        identifier="Art. 1372 c.c.",
        url="https://www.normattiva.it/esempio",
        excerpt="testo",
        score=0.9,
    )
    assert citation.label == "Art. 1372 c.c."


def test_format_citation_matches_expected_layout():
    result = apply_citation_gate([make_passage(0.9)], min_score=0.75)
    formatted = format_citation(result.citations[0])
    assert formatted.startswith("Art. 1372 [Codice Civile]: '")
    assert formatted.endswith("— fonte: https://www.normattiva.it/esempio")
