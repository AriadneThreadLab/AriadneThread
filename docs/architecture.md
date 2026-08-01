# OSM GeoAgent - Architecture

**Knowledge-Grounded Planner-Executor GeoAgent with Tool Orchestration**
(agentic tool orchestration with a bounded tool loop)

The system turns a natural-language GIS request into a small number of
*controlled* OpenStreetMap operations, and returns an answer that can be
verified: the tags it used, the passages it read, the Overpass query it ran,
the features it received, and the order in which all of that happened.

## 1. Request flow

```
User
 ↓
FastAPI endpoint
 ↓
GeoAgent orchestrator
 ↓
Planner (LLM) - does this need OSM documentation? which tags? which area?
 ↓
Tool Registry - validate the requested tool and its arguments
 ↓
Tool execution - search_osm_knowledge and/or query_osm
 ↓
Structured observation appended to the conversation
 ↓
Re-plan (bounded) or finish
 ↓
Answer + sources + execution trace + GeoJSON
```

Planning and execution are not separate services: the planner is the model
turn that emits tool calls, and the executor is the registry turn that runs
them. What makes it a planner-executor loop is that the model never touches an
external system directly - it can only propose a *validated* tool call.

## 2. The central distinction: documentation vs. live data

These two capabilities answer different questions and must never be conflated.

| | `search_osm_knowledge` | `query_osm` |
|---|---|---|
| Source | Local OSM documentation corpus (OSM wiki text) | Overpass API |
| Answers | Which tags model a concept, what `natural=wood` means, Overpass concepts | Which features actually exist right now |
| Freshness | Snapshot, taken at ingestion time | Live |
| Output | Passages with title, section, URL, score | Features, GeoJSON, count, generated query |

Three rules are enforced in the prompt, in the tool observations, and in the
response shape:

1. Documentation is never presented as live map data. Every knowledge
   observation says so explicitly.
2. Live results are never invented. An empty Overpass result is reported as
   empty, with an instruction not to fabricate.
3. `geojson` in the response is populated **only** by `query_osm`.

The typical grounded run for "Find public parks in Berlin" is therefore:
search the documentation to learn that a public park is `leisure=park`, then
query Overpass for `leisure=park` in Berlin, then answer citing both.

## 3. Safe Overpass design

Model-authored Overpass QL is never executed. Instead:

```
Natural language
 ↓
LLM structured tool arguments (JSON)
 ↓
Pydantic validation (OsmFeatureQuery)
 ↓
Deterministic Overpass query builder
 ↓
Overpass API
```

`OsmFeatureQuery` (`app/osm/query_spec.py`) is the security boundary:

- exactly one spatial scope: `place`, `point` (lat/lon/radius) or `bbox`;
- one to eight tag conditions, keys matched against an OSM-key pattern;
- values containing `"`, `\` or newlines are **rejected**, not escaped, so no
  value can alter the structure of the generated query;
- `limit` and `radius_m` are bounded in the schema and clamped again against
  `OVERPASS_MAX_RESULTS` at execution time;
- `extra="forbid"`, so a hallucinated `overpass_ql` field is a validation error
  rather than a silently ignored one.

The builder (`app/osm/query_builder.py`) is pure and deterministic: identical
input always produces an identical query string, which is returned to the user
as provenance.

The supported subset is `[out:json]` + optional `area[name=...]` + node/way/
relation statements with tag selectors and one spatial filter + `out geom|center
<limit>`. Modelling more of Overpass is explicitly out of scope.

The HTTP client (`app/osm/client.py`) posts that query to the configured
`OVERPASS_BASE_URL` only. Connect/read timeouts, `OVERPASS_MAX_RESPONSE_BYTES`,
HTTP 429/4xx/5xx, malformed JSON and Overpass error remarks are mapped to
structured tool errors. Empty `elements` is a valid success.

GeoJSON normalisation (`app/osm/geojson.py`) produces a
`FeatureCollection`. Nodes become Points; ways use `geometry` (LineString or
closed Polygon) or `center` (Point). Relations without reliable geometry are
skipped with an explicit warning - geometry is never invented. Properties keep
`osm_type`, `osm_id`, `tags` and OSM attribution.

`query_osm` returns two layers of result:

1. a compact observation for the model (status, feature count, source type,
   warning count, query summary; never the full GeoJSON);
2. a structured payload for the API/UI (GeoJSON, generated Overpass QL,
   warnings, attribution).

## 4. Bounded tool loop

Implemented by `PlannerExecutorAgent` (`app/agent/orchestrator.py`):

1. Send the conversation and the tool definitions to the model.
2. Receive zero or more tool calls (prompted JSON protocol by default).
3. Validate each call against the registry (tool exists, arguments match).
4. Execute only registered tools.
5. Append a compact structured observation per call (never full GeoJSON).
6. Accumulate structured payloads (`ResultAccumulator`) for the API/UI.
7. Call the model again.
8. Stop on a final answer, or when a limit is reached.

Limits (`app/agent/loop.py`) are `AGENT_MAX_TOOL_ROUNDS` and
`AGENT_MAX_TOOL_CALLS`; per-request network timeouts and response size caps
apply on top. Tool failures are *not* exceptions to the loop - they become
structured observations (`tool_not_registered`, `tool_argument_error`,
`tool_timeout`, ...) so the model can correct itself within the budget.

The production HTTP endpoint is deferred; developers can run one request with
`python -m app.cli agent-query --message "..."`.

## 5. Traceability and hidden reasoning

The trace contains operational events only: `request_received`, `llm_turn`,
`tool_call`, `tool_result`, `tool_error`, `final_answer`, `stopped`.

The installed model emits `<think>...</think>` reasoning. It is stripped at the
provider boundary (`app/llm/tool_protocol.py`) before anything else sees the
reply, and is never traced, returned or persisted.

## 6. Provider boundaries

**LLM.** Domain code depends only on `ChatMessage`, `ToolDefinition`,
`ToolCall`, `LLMResponse` and the `LLMProvider` protocol. `OllamaProvider` is
the only module that knows Ollama's HTTP shape.

`/api/tags` for the configured model reports `completion` and `thinking`
without advertising `tools` (prompted mode is therefore the safe default). The
provider supports two tool-calling modes that share the same tool contracts:

- `prompted` (default): tool schemas are rendered into the system prompt and
  the model replies with `{"tool_calls": [...]}` or `{"final_answer": "..."}`;
- `native`: Ollama's `tools` field, for models that support it.

Switching modes, or adding an OpenAI-compatible cloud provider later, requires
no change to any tool or to the agent.

**Embeddings.** `EmbeddingProvider` / `BgeM3EmbeddingProvider`
(`BAAI/bge-m3`, 1024-dim, multilingual - this is what makes a Persian question
retrieve English OSM documentation). The model is loaded lazily behind a lock
on first use from the local Hugging Face cache only (`local_files_only`);
importing the module pulls in neither torch nor the network, and
`sentence-transformers` is an optional install extra. Indexing is
`python -m app.cli index-osm-knowledge`. Retrieval uses pgvector cosine
distance converted to cosine *similarity* (`score = 1 - distance`; higher is
better), always filtered to `osm_knowledge`, and is exposed as the registered
tool `search_osm_knowledge`.

## 7. Persistence

A dedicated database (`osm_geoagent`) on the existing PostgreSQL 17 server,
reached through `DATABASE_URL` with `postgresql+asyncpg`. No other project's
database or tables are touched.

Tables (see `app/db/models.py`, migration `20260801_0001`):

- `knowledge_documents` - `id`, `domain` (`osm_knowledge`), `title`,
  `source_url`, `source_type`, `license`, `retrieved_at`, `content_hash`,
  `created_at`, `updated_at`. Identity for idempotent ingestion is
  `UNIQUE(domain, source_url)`.
- `knowledge_chunks` - `id`, `document_id`, `section`, `content`,
  `chunk_index`, `content_hash`, `embedding VECTOR(1024)` (nullable until
  indexing), `created_at`, `updated_at`. Position uniqueness is
  `UNIQUE(document_id, chunk_index)` so a refresh can replace orphans cleanly.

The migration enables `postgis` and `vector` with `CREATE EXTENSION IF NOT
EXISTS`. No spatial feature tables are created in the MVP: live GeoJSON remains
request-scoped. No model reasoning is stored.

Migrations are Alembic, run through the same async driver as the application;
`alembic/env.py` reads the URL from settings so no credential lives in a
committed file.

## 8. Knowledge corpus

An explicit whitelist of eight OSM wiki pages (`app/rag/corpus.py`): Map
Features, Tag, `leisure=park`, `landuse=grass`, `natural=wood`,
`landuse=forest`, Overpass QL, Overpass API by Example. There is no crawler and
no link following - `is_allowed_source()` gates ingestion, and extending the
corpus is a deliberate code change. The domain is `osm_knowledge`; these are
*OSM documentation* and *OSM tagging conventions*.

Ingestion (`python -m app.cli ingest-osm-knowledge`) uses the MediaWiki API
only, with a descriptive User-Agent, explicit timeouts and bounded retries.
Articles are normalised from wikitext, split by headings, hashed, and upserted
by `(domain, source_url)`. Changed documents replace all chunks so superseded
passages are not left searchable. Embeddings are written by
`python -m app.cli index-osm-knowledge`.

## 9. Module map

```
app/
  api/          FastAPI routes and dependencies (health)
  agent/        contracts, loop bounds, accumulator, orchestrator, prompt
  core/         settings, logging, error hierarchy
  db/           declarative base, async engine and session lifecycle
  embeddings/   EmbeddingProvider protocol, lazy BGE-M3 provider
  llm/          provider-neutral chat contracts, Ollama provider, tool protocol
  osm/          query spec, builder, Overpass client, GeoJSON encoder
  rag/          retrieval contracts, whitelist, MediaWiki ingest, chunking
  tools/        Tool protocol, registry, search_osm_knowledge, query_osm
  cli.py        ingest/index/search + developer `agent-query`
```

Dependency direction is inward: `api` → `agent` → `tools` → (`rag`, `osm`) →
(`llm`, `embeddings`, `db`) → `core`. Nothing in `tools`, `rag` or `osm`
imports FastAPI or Ollama types.

## 10. Deliberate non-choices

No second vector database (pgvector only), no Qdrant/Pinecone/Milvus/
Elasticsearch, no Kafka/Celery/Kubernetes, no MCP, no workflow engine, no
microservices, and no LangChain/LangGraph: the loop is roughly a hundred lines
of explicit, testable control flow, and a framework would obscure exactly the
part that needs to be auditable.
