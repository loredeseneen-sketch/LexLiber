"""Entry point per l'avvio locale: `uvicorn main:app --reload --port 8000`.

In produzione (docker-compose.yml) il container `api` avvia direttamente
`api.routes:app`; questo modulo esiste solo per comodità di sviluppo locale,
così il comando di avvio non richiede di conoscere il percorso interno del
package. La app FastAPI (incluso il fallback automatico su Qdrant embedded
se Docker non è disponibile, vedi search/retriever.py:build_retriever) è
definita in api/routes.py.
"""
from api.routes import app

__all__ = ["app"]
