# Usage

Install and run the service with [deployment.md](deployment.md). This page
covers the public HTTP API and the offline test suite.

## API

The agent accepts a natural-language message:

| URL | Purpose |
|---|---|
| `http://127.0.0.1:8100/` | Web interface |
| `http://127.0.0.1:8100/api/v1/agent/query` | Agent API (`POST`) |
| `http://127.0.0.1:8100/api/v1/agent/feedback` | Optional feedback for review (`POST`) |
| `http://127.0.0.1:8100/health` | Liveness |
| `http://127.0.0.1:8100/ready` | Readiness |
| `http://127.0.0.1:8100/docs` | OpenAPI |

Example request:

```bash
curl -s http://127.0.0.1:8100/api/v1/agent/query \
  -H 'Content-Type: application/json' \
  -d '{"message":"Find public parks around Istanbul Technical University. Return at most 20 features."}'
```

When spatial analytics runs, the response includes an `analysis` object
(`plan`, `decision_trace`, `result`, `comparison`, `report`). Documentation-only
answers leave `analysis` null.

Configuration is documented in `.env.example`. Important variables include
`DATABASE_URL`, `LLM_PROVIDER`, model endpoints, Overpass limits, and
`AGENT_MAX_TOOL_ROUNDS` / `AGENT_MAX_TOOL_CALLS`.

## Tests

Default tests are offline (no PostgreSQL, Ollama, Overpass, or model download):

```bash
./.venv/bin/pytest
./.venv/bin/ruff check .
./.venv/bin/ruff format --check .
./.venv/bin/mypy app tests
```

The fine-tuning package is validated separately:

```bash
cd finetuning && ../.venv/bin/python -m pytest && ../.venv/bin/mypy src
```

Optional live checks (explicit; not part of the default suite):

```bash
RUN_INTEGRATION=1 ./.venv/bin/pytest -m integration
```

## Safety boundaries

- Documentation is not map data. Only `query_osm` (OSM) or `simbench_query`
  (SimBench) may populate GeoJSON.
- Empty Overpass results stay empty.
- The model does not author Overpass QL; `OsmFeatureQuery` is compiled by
  `build_overpass_query`.
- Tools run only through the Tool Registry with Pydantic validation.
- Hidden `<think>` content is stripped at the provider boundary.
- Agent loops are bounded.
- Active learning never trains a model and never fails a user request.

Map data © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright)
(ODbL). OSM Wiki text is CC-BY-SA.
