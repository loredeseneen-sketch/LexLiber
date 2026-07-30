# LexLiber — Ricerca Giuridica Italiana Sovereign

Sistema open source di ricerca giuridica italiana, self-hostabile, che gira
interamente in locale sul server dello studio legale. Zero dati verso
l'esterno per architettura, non per policy.

## Principi non negoziabili

1. **SOVEREIGN**: nessuna chiamata esterna tranne Claude API (sostituibile
   con modello locale Ollama tramite `LLM_PROVIDER=ollama`). Il corpus
   normativo/giurisprudenziale vive esclusivamente su Qdrant locale.
2. **CITATION**: zero output senza fonte citata. Se il retriever non trova
   passaggi con score > `CITATION_MIN_SCORE` (default 0.75), il sistema
   risponde: *"Non ho trovato fonti sufficienti per rispondere con
   certezza."* — mai inventare (vedi [chat/citation_guard.py](search/citation.py)).
3. **SEMPLICE**: un avvocato non tecnico deve poterlo usare. Un solo comando
   (`docker compose up`) per l'installazione. Frontend HTML + vanilla JS,
   zero build step, zero framework.

## Cosa NON fare mai

- Chiamate API esterne per recuperare documenti giuridici a runtime (lo
  scraping avviene offline/batch in `corpus/`, popola Qdrant una volta,
  poi tutto gira in locale).
- Restituire testo generato dal LLM come risposta finale senza allegare
  almeno una citazione con fonte + link + estratto.
- Aggiungere dipendenze che richiedono servizi cloud esterni (no Pinecone,
  no OpenAI embeddings API, no hosted vector DB).
- Complicare il frontend con framework, bundler o build step.

## Architettura

```
Utente → frontend/index.html → api/routes.py (FastAPI)
                                     │
                        ┌────────────┴────────────┐
                        ▼                          ▼
              search/retriever.py         chat/engine.py
                  (Qdrant query)          (Claude API / Ollama)
                        │                          │
              search/embedder.py          chat/prompts.py
           (sentence-transformers)                 │
                        │                          ▼
                        └──────────► search/citation.py
                                     (enforce citation gate)
```

Il corpus (Normattiva, EUR-Lex, Cassazione pubblica) viene scaricato
offline dagli scraper in `corpus/scrapers/`, normalizzato e indicizzato da
`corpus/ingest.py` dentro Qdrant. A runtime il sistema non contatta più
alcuna fonte esterna: interroga solo Qdrant locale e (opzionalmente)
l'API Claude per generare la risposta in linguaggio naturale a partire
dai passaggi recuperati.

## Comandi utili

```bash
cp .env.example .env               # configura ANTHROPIC_API_KEY o Ollama
docker compose up -d               # avvia Qdrant + API + frontend
python corpus/ingest.py --source normattiva --query "codice civile"
pytest tests/ -v
```

## Stack

- Backend: Python 3.11 + FastAPI
- Vector store: Qdrant (Docker, locale, nessun account cloud)
- Embeddings: `sentence-transformers` multilingue (locale, no API)
- LLM: Claude API di default, fallback Ollama (`LLM_PROVIDER=ollama`)
- Frontend: HTML + vanilla JS, nessun framework
- Corpus: Normattiva, EUR-Lex, Corte di Cassazione (fonti pubbliche)
- Deploy: `docker-compose.yml`, un comando solo
