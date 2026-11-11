# Retrieval-Augmented Generation

Ariadne Thread grounds tagging answers in a local OpenStreetMap Wiki corpus and
grounds solar-PV methodology answers in a second, separate corpus of packaged
notes. The language model does not decide OSM tags or solar regulatory meaning
from parametric memory alone: it reads retrieved passages first.

## Why RAG is used

Geographic tagging conventions are specific and change over time. Photovoltaic
methodology, installation practice, grid codes, and permitting language are
likewise document-backed. A planner that relies only on training data can invent
tags or invent site indicators. Retrieval supplies citable documentation before
a live query or a SolarDSS routing call.

RAG answers questions *about* OpenStreetMap and *about* solar-PV knowledge. It
never returns map features. It never returns SolarDSS indicator values. Live
geometries come from Overpass or SimBench. Deterministic solar indicators come
from SolarDSS.

## Corpus

Ingestion is restricted to explicit whitelists in `app/rag/corpus.py`. There is
no crawler and no link following. Adding a page is a deliberate code change.

The OSM whitelist includes map-feature and tag pages (parks, grass, wood,
forest) and Overpass QL documentation. Wiki text is licensed CC-BY-SA.

The solar-PV whitelist is packaged markdown under `app/rag/solar_documents/`.
It covers PV methodology, panel documentation, installation guidelines, grid
rules, permitting rules, regulatory documents, technical standards, and
scientific methodology. Those notes **must not** contain scientific numeric
calculations. Typical questions:

- Why was this site excluded?
- What does GHI mean?
- Why is slope important?
- What regulation affects this site?

## Retrieval pipeline

1. Approved documents are ingested and split into heading-aware chunks.
2. Chunks are embedded with BGE-M3 and stored in PostgreSQL using pgvector.
3. `search_osm_knowledge` and `search_solar_knowledge` embed the user query and
   return the top matching passages (title, section, URL, score, domain).
4. The planner sees a compact observation. The API/UI receive citations as
   `knowledge_sources` and a partitioned `evidence` object.

The embedding model is an optional install extra. The application does not
download it implicitly.

## Grounding rule

Documentation, live OSM data, and SolarDSS numbers stay separate:

| Source | What it is | What it is not |
|---|---|---|
| `search_osm_knowledge` | OSM Wiki passages and citations | Map features |
| `search_solar_knowledge` | Solar-PV knowledge / document evidence | SolarDSS indicators |
| `query_osm` | Live GeoJSON from Overpass | Documentation |
| `analyze_solar_site` | Deterministic analytical routing slots | RAG-generated GHI/slope |

An answer may cite both evidence classes. Only `query_osm` (OSM),
`simbench_query` (SimBench), or `analyze_energy_grid` may populate `geojson`.
Never use RAG-generated values as replacements for deterministic SolarDSS
indicators.

The public response keeps two labelled lists:

- **DETERMINISTIC ANALYTICAL EVIDENCE** — SolarDSS routing, spatial analytics,
  GeoLoadST. Citations and provenance stay on the engine item.
- **KNOWLEDGE / DOCUMENT EVIDENCE** — retrieved passages with title, section,
  URL, score, domain, and document kind.
