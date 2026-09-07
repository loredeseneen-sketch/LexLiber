# LexLiber

Motore di ricerca giuridica italiana self-hosted: ricerca semantica e
lookup esatto per articolo su un corpus normativo indicizzato localmente
(Qdrant), con citazioni obbligatorie e una sezione per analizzare
documenti caricati (PDF/DOCX) alla luce delle norme applicabili.

Vedi [CLAUDE.md](CLAUDE.md) per i principi architetturali (SOVEREIGN,
CITATION, SEMPLICE) a cui il progetto si attiene.

## Prerequisiti

- Python 3.11+
- Docker Desktop (consigliato, per Qdrant containerizzato) — oppure nessun
  Docker: il sistema ricade automaticamente su un'istanza Qdrant embedded
  su file locale (`data/qdrant_local/`)
- Una API key Anthropic (`ANTHROPIC_API_KEY`) — oppure [Ollama](https://ollama.com)
  installato in locale per non dipendere da alcun servizio cloud
  (`LLM_PROVIDER=ollama`)

## Setup

### 1. Clona e configura l'ambiente

```bash
git clone https://github.com/loredeseneen-sketch/LexLiber.git
cd LexLiber

python3 -m venv .venv
source .venv/bin/activate        # su Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configura le variabili d'ambiente

```bash
cp .env.example .env
```

Apri `.env` e imposta almeno:

```
LLM_PROVIDER=claude
ANTHROPIC_API_KEY=sk-ant-...      # la tua vera key, MAI committata (vedi .gitignore)
```

In alternativa, per restare 100% locale senza alcuna chiamata esterna:

```
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1
```

`.env` è escluso da git (vedi `.gitignore`): non va mai committato.

### 3. Popola il corpus (Qdrant)

Il corpus va costruito una volta offline, scaricando e indicizzando i
codici normativi (Normattiva) e/o gli atti EUR-Lex:

```bash
# Ingestione completa (tutti i codici Normattiva)
python corpus/ingest.py --source normattiva

# Solo alcuni codici, con limite di articoli (utile per un test rapido)
python corpus/ingest.py --source normattiva --codici codice_civile --limit 50

# EUR-Lex
python corpus/ingest.py --source eurlex --limit 5

# Dry-run: valida lo scraping senza scrivere su Qdrant
python corpus/ingest.py --source all --dry-run
```

Questo è l'unico punto del sistema che contatta fonti esterne (Normattiva,
EUR-Lex): va eseguito manualmente/offline. A runtime l'API non fa mai
chiamate di rete verso l'esterno, a parte l'eventuale generazione LLM.

### 4. Avvia il backend

**Con Docker (consigliato)** — avvia Qdrant + API + frontend con un solo comando:

```bash
docker compose up -d
```

- API disponibile su `http://127.0.0.1:8000`
- Frontend su `http://127.0.0.1:3000`
- Qdrant su `127.0.0.1:6333` (solo localhost, non esposto all'esterno)

**Senza Docker** — l'API ricade automaticamente su Qdrant embedded su file
locale se non trova un'istanza Qdrant raggiungibile:

```bash
uvicorn main:app --reload --port 8000
```

### 5. Apri il frontend

Se non usi Docker, servi `frontend/index.html` con un semplice server
statico (aprirlo direttamente come file `file://` funziona comunque, ma un
server evita eventuali restrizioni CORS del browser):

```bash
cd frontend && python3 -m http.server 3000
```

Poi apri `http://localhost:3000` nel browser. Se l'API non gira sulla porta
di default, imposta `window.LEXLIBER_API_BASE` in una riga `<script>` prima
del caricamento di `index.html`.

## Verifica installazione

```bash
curl http://127.0.0.1:8000/api/health
# {"status": "ok"}

curl http://127.0.0.1:8000/health
# {"status": "ok", "chunk_totali": N, "ultimo_aggiornamento": "..."}
```

## Test

```bash
pytest tests/ -v
```

## Struttura del progetto

```
frontend/index.html      Frontend statico (HTML + CSS + JS, zero dipendenze)
api/routes.py             Endpoint FastAPI (/search, /analyze-document, /health)
search/retriever.py       Ricerca semantica + lookup esatto per articolo su Qdrant
search/citation.py        Citation gate: nessuna risposta LLM senza fonte verificata
search/document_extractor.py  Estrazione testo locale da PDF/DOCX caricati
chat/engine.py             Orchestrazione retrieval → citation gate → LLM
corpus/ingest.py           Pipeline offline di scraping + indicizzazione
corpus/scrapers/           Scraper per Normattiva, EUR-Lex, Cassazione
```

## Stack

- Backend: Python 3.11 + FastAPI
- Vector store: Qdrant (Docker locale, o embedded su file — nessun account cloud)
- Embeddings: `sentence-transformers` multilingue (locale)
- Reranking: cross-encoder locale (`search/reranker.py`)
- LLM: Claude API di default, fallback Ollama 100% locale
- Frontend: HTML + vanilla JS, nessun framework/build step
