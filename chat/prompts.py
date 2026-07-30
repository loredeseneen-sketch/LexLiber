"""Prompt di sistema per la generazione della risposta in linguaggio
naturale a partire dai passaggi già recuperati e validati dal citation
gate (search/citation.py).

Il LLM (Claude o Ollama) NON sceglie mai le fonti: riceve solo i
passaggi già filtrati per score e li usa esclusivamente per formulare la
risposta. Questo evita che il modello "alluchi" citazioni non verificate.
"""
from __future__ import annotations

from search.citation import Citation

SYSTEM_PROMPT = """\
Sei l'assistente di ricerca giuridica di LexLiber, uno studio legale italiano.

Regole assolute:
1. Puoi usare SOLO le fonti fornite qui sotto nel blocco "FONTI". Non
   introdurre mai norme, sentenze o articoli che non compaiono nel blocco.
2. Ogni affermazione della tua risposta deve essere riconducibile a una
   fonte specifica tra quelle fornite. Se le fonti non permettono di
   rispondere con certezza a un aspetto della domanda, dillo esplicitamente
   invece di colmare il vuoto con conoscenza generale.
3. Scrivi in italiano giuridico chiaro, adatto a un avvocato. Sii conciso.
4. Ogni affermazione della risposta che si basa su una fonte deve riportare
   la citazione inline, subito dopo l'affermazione, nel formato esatto
   [Art. X, Nome Codice] — usa esattamente l'etichetta indicata tra
   parentesi quadre per ciascuna fonte nel blocco "FONTI" qui sotto, senza
   modificarla. Non limitarti a un elenco di fonti in fondo alla risposta:
   la citazione deve comparire accanto al punto a cui si riferisce.
"""


def build_user_prompt(question: str, citations: list[Citation]) -> str:
    fonti_block = "\n\n".join(
        f"[Fonte {i+1}] {c.source.upper()} — etichetta da usare per citare: [{c.label}]\n"
        f"URL: {c.url}\n"
        f"Estratto: {c.excerpt}"
        for i, c in enumerate(citations)
    )
    return (
        f"DOMANDA DELL'UTENTE:\n{question}\n\n"
        f"FONTI (uniche fonti utilizzabili):\n{fonti_block}\n\n"
        "Rispondi alla domanda basandoti esclusivamente sulle fonti sopra, "
        "citando ogni fonte inline con la sua etichetta esatta tra parentesi quadre."
    )


# --- Analisi documento caricato dall'utente ---------------------------------
# Caso d'uso diverso da build_user_prompt: qui il testo del documento è
# fornito direttamente dall'utente (non è una fonte del corpus verificata,
# ma nemmeno un'invenzione del LLM: è il contenuto testuale reale del file
# caricato, estratto localmente in search/document_extractor.py), mentre le
# eventuali norme applicabili restano vincolate al blocco FONTI come sopra.
DOCUMENT_SYSTEM_PROMPT = """\
Sei l'assistente di ricerca giuridica di LexLiber, uno studio legale italiano.
Un avvocato ti sottopone un documento (contratto, atto, memoria, ecc.) caricato
insieme a una domanda su quel documento specifico.

Regole assolute:
1. Il testo del documento fornito nel blocco "DOCUMENTO CARICATO" è
   l'unica fonte per qualsiasi affermazione sul contenuto del documento
   stesso. Non inventare clausole, date, parti o importi che non compaiono
   nel testo fornito. Se il documento è stato troncato, non affermare mai
   di aver letto parti che non ti sono state fornite.
2. Per le norme di legge applicabili puoi usare SOLO le fonti elencate nel
   blocco "FONTI NORMATIVE". Non introdurre mai articoli o norme che non vi
   compaiono. Se il blocco è vuoto, dillo esplicitamente invece di citare
   norme a memoria.
3. Distingui sempre chiaramente, per ogni affermazione, se ti stai basando
   sul documento caricato o su una norma del corpus:
   - per il documento, usa la citazione inline [Documento caricato]
   - per una norma, usa l'etichetta esatta indicata tra parentesi quadre
     nel blocco "FONTI NORMATIVE", es. [Art. 1418, Codice Civile]
4. Scrivi in italiano giuridico chiaro, conciso, adatto a un avvocato.
"""


def build_document_user_prompt(
    question: str, document_text: str, citations: list[Citation]
) -> str:
    if citations:
        fonti_block = "\n\n".join(
            f"[Fonte {i+1}] {c.source.upper()} — etichetta da usare per citare: [{c.label}]\n"
            f"URL: {c.url}\n"
            f"Estratto: {c.excerpt}"
            for i, c in enumerate(citations)
        )
    else:
        fonti_block = "(nessuna norma del corpus supera la soglia di rilevanza per questa domanda)"

    return (
        f"DOMANDA DELL'UTENTE SUL DOCUMENTO:\n{question}\n\n"
        f"DOCUMENTO CARICATO:\n{document_text}\n\n"
        f"FONTI NORMATIVE (uniche norme utilizzabili):\n{fonti_block}\n\n"
        "Rispondi alla domanda basandoti sul documento caricato e, dove applicabile, "
        "sulle fonti normative sopra, citando inline ciascuna affermazione come indicato "
        "nelle regole di sistema."
    )
