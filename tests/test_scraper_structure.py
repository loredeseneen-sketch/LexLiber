"""Test degli helper di parsing condivisi tra gli scraper (nessuna
chiamata di rete: sono funzioni pure su testo sintetico)."""
from corpus.models import ArticleChunk
from corpus.scrapers.structure import extract_structure_labels, split_articles_from_text


def test_extract_structure_labels_finds_last_occurrence_of_each_level():
    text = (
        "LIBRO QUARTO - Delle obbligazioni\n"
        "TITOLO II - Dei contratti in generale\n"
        "CAPO XIV - Della risoluzione del contratto\n"
        "Art. 1453 (Risolubilità del contratto per inadempimento)\n"
        "Nei contratti con prestazioni corrispettive..."
    )
    result = extract_structure_labels(text)
    assert result["libro"].startswith("LIBRO QUARTO")
    assert result["titolo"].startswith("TITOLO II")
    assert result["capo"].startswith("CAPO XIV")


def test_extract_structure_labels_handles_parte_as_libro():
    text = "PARTE PRIMA - Disposizioni generali\nArt. 1\nTesto."
    result = extract_structure_labels(text)
    assert result["libro"].startswith("PARTE PRIMA")
    assert result["titolo"] is None
    assert result["capo"] is None


def test_extract_structure_labels_returns_none_when_absent():
    result = extract_structure_labels("Nessuna struttura qui, solo testo.")
    assert result == {"libro": None, "titolo": None, "capo": None}


def test_split_articles_from_text_splits_on_each_header():
    text = (
        "Articolo 1\n"
        "Oggetto e finalità del regolamento.\n"
        "Articolo 2\n"
        "Definizioni utilizzate nel presente atto.\n"
        "Articolo 3\n"
        "Ambito di applicazione."
    )
    articles = split_articles_from_text(text)
    assert [a[0] for a in articles] == ["1", "2", "3"]
    assert articles[0][2] == "Oggetto e finalità del regolamento."
    assert "Ambito di applicazione" in articles[2][2]


def test_split_articles_from_text_ignores_empty_trailing_header():
    text = "Articolo 1\nTesto vero.\nArticolo 2\n"
    articles = split_articles_from_text(text)
    # "Articolo 2" senza testo successivo non produce una voce.
    assert [a[0] for a in articles] == ["1"]


def test_article_chunk_id_is_stable_and_deterministic():
    chunk = ArticleChunk(
        fonte="normattiva",
        codice="Codice Civile",
        libro="LIBRO IV",
        titolo="TITOLO II",
        capo="CAPO XIV",
        articolo="Art. 1453",
        numero_articolo="1453",
        testo="Nei contratti con prestazioni corrispettive...",
        url_originale="https://www.normattiva.it/x~art1453",
    )
    same = ArticleChunk(**{**chunk.__dict__})
    assert chunk.chunk_id == same.chunk_id

    payload = chunk.to_payload()
    assert payload["chunk_id"] == chunk.chunk_id
    assert payload["numero_articolo"] == "1453"
    assert payload["url_originale"] == "https://www.normattiva.it/x~art1453"
