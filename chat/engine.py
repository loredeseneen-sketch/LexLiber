"""Motore di chat: orchestra retrieval + citation gate + generazione LLM.

Flusso per ogni domanda (vedi CLAUDE.md):
1. `Retriever.search` interroga Qdrant locale (nessuna chiamata esterna).
2. `apply_citation_gate` filtra i passaggi per score >= soglia. Se non
   sopravvive nessun passaggio, si esce subito con il messaggio di
   fallback fisso: MAI si chiama il LLM senza fonti valide.
3. Solo se ci sono fonti sufficienti, il LLM (Claude o Ollama, a seconda
   di LLM_PROVIDER) genera la sintesi in linguaggio naturale, vincolato a
   usare solo le fonti fornite.
4. La risposta finale include sempre la sintesi + l'elenco delle citazioni
   strutturate (fonte, url, estratto).
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from chat.prompts import (
    DOCUMENT_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_document_user_prompt,
    build_user_prompt,
)
from config import settings
from search.citation import Citation, apply_citation_gate
from search.retriever import ARTICLE_NOT_FOUND_MESSAGE, Retriever, detect_article_query

logger = logging.getLogger(__name__)


def _init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                sufficient_sources INTEGER NOT NULL,
                citations_json TEXT NOT NULL
            )"""
        )


@dataclass
class ChatAnswer:
    answer: str
    citations: list[Citation]
    sufficient_sources: bool


class ChatEngine:
    def __init__(self, retriever: Retriever | None = None, db_path: str | None = None):
        self.retriever = retriever or Retriever()
        self.db_path = Path(db_path or settings.sqlite_path)
        _init_db(self.db_path)

    def ask(self, question: str) -> ChatAnswer:
        article_ref = detect_article_query(question)
        if article_ref is not None:
            numero, codice = article_ref
            passage = self.retriever.search_by_article(numero, codice)
            if passage is None:
                logger.info(
                    "Lookup esatto articolo: nessun match per numero=%r codice=%r", numero, codice
                )
                answer = ChatAnswer(
                    answer=ARTICLE_NOT_FOUND_MESSAGE, citations=[], sufficient_sources=False
                )
                self._save_interaction(question, answer)
                return answer
            passages = [passage]
        else:
            passages = self.retriever.search(question)

        gate = apply_citation_gate(passages)

        if not gate.sufficient:
            logger.info("Citation gate: fonti insufficienti per %r", question)
            answer = ChatAnswer(
                answer=gate.fallback_message or "",
                citations=[],
                sufficient_sources=False,
            )
            self._save_interaction(question, answer)
            return answer

        answer_text = self._generate(question, gate.citations)
        answer = ChatAnswer(answer=answer_text, citations=gate.citations, sufficient_sources=True)
        self._save_interaction(question, answer)
        return answer

    def analyze_document(self, question: str, document_text: str) -> ChatAnswer:
        """Risponde a una domanda su un documento caricato dall'utente
        (vedi search/document_extractor.py), citando sia il documento sia le
        norme del corpus applicabili trovate dal retriever semantico.

        A differenza di `ask()`, l'assenza di norme del corpus sopra soglia
        NON blocca la generazione: il documento caricato è comunque una
        fonte legittima (testo reale, non inventato) su cui rispondere.
        `sufficient_sources` qui indica solo se sono state trovate norme
        applicabili, non se la risposta nel suo complesso è valida — il
        frontend lo usa per il badge di confidence sulle fonti normative."""
        passages = self.retriever.search(question)
        gate = apply_citation_gate(passages)
        citations = gate.citations if gate.sufficient else []

        answer_text = self._generate(
            question, citations, document_text=document_text, system_prompt=DOCUMENT_SYSTEM_PROMPT
        )
        answer = ChatAnswer(answer=answer_text, citations=citations, sufficient_sources=gate.sufficient)
        self._save_interaction(f"[Documento] {question}", answer)
        return answer

    def _save_interaction(self, question: str, answer: ChatAnswer) -> None:
        """Salva query + risposta + fonti su SQLite locale, per storico e
        auditing (nessun servizio esterno coinvolto). Un errore di
        persistenza non deve mai far fallire la risposta all'utente."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO interactions "
                    "(created_at, question, answer, sufficient_sources, citations_json) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        datetime.now(timezone.utc).isoformat(),
                        question,
                        answer.answer,
                        int(answer.sufficient_sources),
                        json.dumps([c.to_dict() for c in answer.citations], ensure_ascii=False),
                    ),
                )
        except Exception:
            logger.exception("Impossibile salvare l'interazione su SQLite (%s)", self.db_path)

    def _generate(
        self,
        question: str,
        citations: list[Citation],
        document_text: str | None = None,
        system_prompt: str = SYSTEM_PROMPT,
    ) -> str:
        if document_text is not None:
            user_prompt = build_document_user_prompt(question, document_text, citations)
        else:
            user_prompt = build_user_prompt(question, citations)
        if settings.llm_provider == "ollama":
            return self._generate_ollama(user_prompt, system_prompt)
        return self._generate_claude(user_prompt, system_prompt)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def _generate_claude(self, user_prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        message = client.messages.create(
            model=settings.claude_model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(
            block.text for block in message.content if getattr(block, "type", None) == "text"
        ).strip()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def _generate_ollama(self, user_prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        resp = httpx.post(
            f"{settings.ollama_base_url}/api/chat",
            json={
                "model": settings.ollama_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
