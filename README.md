# OSM GeoAgent

Turns natural-language GIS requests into controlled OpenStreetMap operations,
grounded in OSM documentation and answered with real Overpass data.

```
"Find public parks in Berlin."
"فضاهای سبز برلین را از OpenStreetMap پیدا کن."
"Find pharmacies around Alexanderplatz."
"What OSM tags should be used for urban green areas?"
```

The agent looks up OSM tagging conventions in a local documentation corpus,
then queries live features through the Overpass API, and returns an answer with
its sources, an execution trace and GeoJSON.

Architecture: **Knowledge-Grounded Planner-Executor GeoAgent with Tool
Orchestration**. See [docs/architecture.md](docs/architecture.md).

## Status

Foundation, knowledge persistence, and whitelisted OSM wiki ingestion. The
application starts, the health endpoint works, knowledge tables are migrated,
and the corpus can be fetched into PostgreSQL without embeddings yet.
Embedding/retrieval, the Overpass HTTP client and the agent loop are next - see
[Roadmap](#roadmap).

## Requirements

- Python 3.10 (WSL2 Ubuntu 22.04 is the reference environment)
- PostgreSQL 17 with the `postgis` and `vector` extensions
- Ollama with a local chat model (`deepseek-r1:7b` is the verified default)
- Optional, for embeddings: an NVIDIA GPU and the `BAAI/bge-m3` model

## Setup

```bash
python3 -m venv .venv
./.venv/bin/pip install -e ".[dev]"      # add ".[dev,embeddings]" for BGE-M3
cp .env.example .env                      # then fill in DATABASE_URL
```

Create the project's own database (do not reuse another project's):

```bash
createdb -h 127.0.0.1 -p 5433 osm_geoagent
psql -h 127.0.0.1 -p 5433 -d osm_geoagent -c "CREATE EXTENSION IF NOT EXISTS postgis;"
psql -h 127.0.0.1 -p 5433 -d osm_geoagent -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

## Run

```bash
./.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8100 --reload
curl http://127.0.0.1:8100/health
```

Interactive API docs: <http://127.0.0.1:8100/docs>

## Checks

```bash
./.venv/bin/pytest            # no network, no database, no model downloads
./.venv/bin/ruff check .
./.venv/bin/ruff format --check .
./.venv/bin/mypy app tests
```

## Migrations

After creating `.env` with a real `DATABASE_URL` and creating the empty
`osm_geoagent` database:

```bash
./.venv/bin/alembic upgrade head
```

The first revision enables `postgis` and `vector`, then creates the knowledge
tables. The URL comes from `DATABASE_URL`; `alembic.ini` holds no credentials.

## Configuration

All settings live in `.env` (see `.env.example`): application host/port,
`DATABASE_URL`, Ollama base URL/model/context/timeout, BGE-M3 model/device/
batch size, Overpass endpoint/timeout/limits, `RAG_TOP_K`, and the agent loop
bounds `AGENT_MAX_TOOL_ROUNDS` / `AGENT_MAX_TOOL_CALLS`.

## Ingest OSM documentation

Fetch the eight whitelisted OSM wiki pages via the MediaWiki API, chunk them
heading-aware, and upsert into `knowledge_documents` / `knowledge_chunks`
(idempotent; no embeddings in this step):

```bash
./.venv/bin/python -m app.cli ingest-osm-knowledge
```

## Roadmap

1. ~~`knowledge_documents` / `knowledge_chunks` models and the first migration~~
2. ~~OSM wiki ingestion for the eight whitelisted pages, plus chunking~~
3. BGE-M3 indexing and pgvector similarity retrieval
4. The Overpass HTTP client and the Overpass-to-GeoJSON encoder
5. The bounded agent loop and the `/agent/query` endpoint
6. A map UI

## Data and licensing

Map data is © OpenStreetMap contributors, licensed ODbL. OSM wiki text is
CC-BY-SA. Attribution is carried through every `query_osm` result.
