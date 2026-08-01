# AGENTS.md - working rules for this repository

OSM GeoAgent converts natural-language GIS requests into controlled
OpenStreetMap operations. Read [docs/architecture.md](docs/architecture.md)
before making structural changes.

## Repository boundary

- This repository is `/home/rasoul/projects/osm-geoagent`. Never modify another
  repository, in particular `/home/rasoul/projects/legal-ai-platform`, and never
  import from it.
- Shared *infrastructure* (the PostgreSQL server, the Ollama service, the GPU)
  is reused. Everything else is separate: virtualenv `.venv`, database
  `osm_geoagent`, migrations, configuration, tables.

## Environment

- Python 3.10 only. No `match`, no `Self`, no PEP 604 runtime-only syntax that
  3.10 rejects. Use `from __future__ import annotations`.
- Use `./.venv/bin/python`, `./.venv/bin/pytest`, etc. Do not install packages
  globally.
- PostgreSQL 17 listens on `127.0.0.1:5433`. Ollama on `127.0.0.1:11434`.
- Never `ollama pull`, delete, upgrade or replace a model. Discover what is
  installed with `ollama list` and use that tag.
- Never download `BAAI/bge-m3` implicitly; it is already cached under
  `~/.cache/huggingface/hub`.
- Credentials come from `.env` only. Never commit `.env`, never guess database
  credentials, never put a URL with a password in `alembic.ini` or in code.

## Non-negotiable design rules

1. **Documentation is not map data.** `search_osm_knowledge` returns OSM
   documentation; `query_osm` returns live features. Never let an answer, an
   observation or a response field blur the two. `geojson` is populated only by
   `query_osm`.
2. **Never invent OSM results.** Empty results are reported as empty.
3. **No model-authored Overpass QL.** The model fills in `OsmFeatureQuery`;
   `build_overpass_query` renders it. If a new capability is needed, extend the
   spec *and* the builder *and* the tests - never add a raw-query escape hatch.
4. **The registry is the only execution path.** Tools run only if registered and
   only with arguments that validate against their own Pydantic model.
5. **No hidden reasoning anywhere.** `<think>` content is stripped at the
   provider boundary and must never be traced, returned, logged or persisted.
6. **Bounded loops.** Every agent run respects `AGENT_MAX_TOOL_ROUNDS` and
   `AGENT_MAX_TOOL_CALLS`; every outbound call has an explicit timeout.
7. **Corpus is a whitelist.** Ingest only the pages in `app/rag/corpus.py`.
   No crawler, no link following.

## Code conventions

- Strongly typed; `mypy` runs in strict mode over `app`. New code must pass.
- Contracts live in `contracts.py` modules and are `Protocol`s or frozen
  dataclasses/Pydantic models. Implementations depend on contracts, never the
  reverse.
- One schema per concept. `query_osm`'s arguments *are* `OsmFeatureQuery`; do
  not define a parallel model.
- No module-level singletons for engines, clients or models. Long-lived
  resources are created in the FastAPI lifespan and read from `app.state`.
- Nothing at import time may connect to a network, load a model or open a pool.
- Tool failures raise `ToolError` subclasses; the registry converts them into
  structured observations.
- Comments explain constraints and intent, not what the next line does.

## Testing

- `./.venv/bin/pytest` must pass with **no** network, no database, no Ollama and
  no model download. Use `httpx.MockTransport` and hand-written fakes.
- Test behaviour that matters: validation rejections, injection attempts, loop
  bounds, error mapping, exact generated queries.
- Before finishing any change, run:

```bash
./.venv/bin/pytest && ./.venv/bin/ruff check . && ./.venv/bin/ruff format --check . && ./.venv/bin/mypy app tests
```

## Scope discipline

Do not add Qdrant, Pinecone, Milvus, Elasticsearch, Kafka, Celery, Kubernetes,
MCP, a workflow orchestrator, microservices, or LangChain/LangGraph. pgvector is
the only vector store. Prefer deleting an abstraction over adding a speculative
one.
