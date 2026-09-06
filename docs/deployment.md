# Deployment

Ariadne Thread is a single FastAPI process: the JSON API and the web interface
are served together. It is intended for local or lab deployment, not a hardened
multi-tenant service.

## Dependencies

- Python **3.10**
- PostgreSQL **17** with **PostGIS** and **pgvector**
- An LLM provider:
  - Ollama (`LLM_PROVIDER=ollama`) and a locally installed model tag, or
  - an OpenAI-compatible gateway (`LLM_PROVIDER=avalai`) with `AVALAI_API_KEY`
- Optional: BGE-M3 for corpus indexing and retrieval (`pip install -e ".[embeddings]"`)

The application does not download Ollama or Hugging Face models automatically.

## Environment

```bash
python3 -m venv .venv
./.venv/bin/pip install -e ".[dev]"
./.venv/bin/pip install -e ".[embeddings]"   # if you will index or search the corpus
./.venv/bin/pip install -e ".[energy]"       # SimBench data tool
./.venv/bin/pip install -e ../AriadneThread-GeoLoadST  # analyze_energy_grid plugin
cp .env.example .env
```

Set a real `DATABASE_URL` in `.env`. Never commit `.env`. See `.env.example` for
LLM, Overpass, Nominatim, and agent-budget variables.

## Database

```bash
createdb osm_geoagent
psql -d osm_geoagent -c "CREATE EXTENSION IF NOT EXISTS postgis;"
psql -d osm_geoagent -c "CREATE EXTENSION IF NOT EXISTS vector;"
./.venv/bin/alembic upgrade head
```

Index the whitelisted OSM Wiki corpus (required for RAG):

```bash
./.venv/bin/python -m app.cli ingest-osm-knowledge
./.venv/bin/python -m app.cli index-osm-knowledge
```

## Running the backend and frontend

```bash
./.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8100 --reload
```

| URL | Role |
|---|---|
| `http://127.0.0.1:8100/` | Web interface (query form, workflow, map) |
| `http://127.0.0.1:8100/api/v1/agent/query` | Agent API |
| `http://127.0.0.1:8100/api/v1/agent/feedback` | Optional feedback |
| `http://127.0.0.1:8100/health` | Liveness |
| `http://127.0.0.1:8100/ready` | Readiness (database and LLM provider) |
| `http://127.0.0.1:8100/docs` | OpenAPI |

The map uses MapLibre and OpenFreeMap; the browser needs network access for
tiles and fonts. There is no separate frontend build step.

## Tests

Default tests are offline (no database, Ollama, Overpass, or model download):

```bash
./.venv/bin/pytest
./.venv/bin/ruff check .
./.venv/bin/ruff format --check .
./.venv/bin/mypy app tests
```

Optional live checks: `RUN_INTEGRATION=1 ./.venv/bin/pytest -m integration`.

API examples: [usage.md](usage.md).
