"""Configurazione centralizzata, letta da variabili d'ambiente (.env).

Nessun default punta a servizi cloud esterni oltre a Claude API, che è
comunque disattivabile impostando LLM_PROVIDER=ollama (vedi CLAUDE.md).
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    llm_provider: str = "claude"
    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-4-6"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "lexliber_corpus"

    # Embeddings
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    embedding_dim: int = 768

    # Citation gate (CRITICA, vedi CLAUDE.md)
    citation_min_score: float = 0.75
    citation_max_excerpt_lines: int = 3
    citation_top_k: int = 5

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Storico conversazioni (SQLite locale, nessun servizio esterno)
    sqlite_path: str = "./data/lexliber.db"

    # Analisi documenti caricati (PDF/DOCX, vedi search/document_extractor.py).
    # L'estrazione testo è sempre locale (pypdf/python-docx), coerente con
    # SOVEREIGN: il file non lascia mai il server, solo l'eventuale sintesi
    # LLM finale passa per Claude API (o Ollama).
    document_max_size_mb: int = 20
    # ~40.000 caratteri coprono comodamente un atto di alcune decine di
    # pagine restando entro una dimensione di prompt ragionevole; oltre la
    # soglia il testo viene troncato (con avviso esplicito nel prompt, così
    # il LLM non finge di aver letto l'intero documento).
    document_max_chars: int = 40000

    # Reranking locale (cross-encoder, vedi search/reranker.py). Affina il
    # ranking del bi-encoder su un pool di candidati più ampio del top-k
    # finale; se il modello non è disponibile si ricade sul cosine score.
    rerank_enabled: bool = True
    rerank_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    rerank_candidate_multiplier: int = 5
    rerank_min_candidates: int = 20


settings = Settings()
