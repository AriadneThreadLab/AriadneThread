# Retrieval-Augmented Generation

Ariadne Thread grounds tagging and documentation answers in a local OpenStreetMap
Wiki corpus. The language model does not decide OSM tags from parametric memory
alone: it reads retrieved passages first.

## Why RAG is used

Geographic tagging conventions are specific and change over time. A planner that
relies only on its training data can invent tags or confuse related concepts
(for example `leisure=park` versus `landuse=grass`). Retrieval supplies the
current, citable OSM documentation before a live query is issued.

RAG in this project answers questions *about* OpenStreetMap. It never returns
map features. Live geometries come only from the Overpass tool.

## Corpus

Ingestion is restricted to an explicit whitelist (`app/rag/corpus.py`). There is
no crawler and no link following. Adding a page is a deliberate code change.

The current whitelist includes OSM map-feature and tag pages (parks, grass,
wood, forest) and Overpass QL documentation. Wiki text is licensed CC-BY-SA.

## Retrieval pipeline

1. Approved pages are ingested and split into heading-aware chunks.
2. Chunks are embedded with BGE-M3 and stored in PostgreSQL using pgvector.
3. `search_osm_knowledge` embeds the user query and returns the top matching
   passages (title, section, URL, score).
4. The planner sees a compact observation. The API/UI receive the citations as
   `knowledge_sources`.

The embedding model is an optional install extra. The application does not
download it implicitly.

## Grounding rule

Documentation and live OSM data stay separate:

| Source | What it is | What it is not |
|---|---|---|
| `search_osm_knowledge` | OSM Wiki passages and citations | Map features |
| `query_osm` | Live GeoJSON from Overpass | Documentation |

An answer may cite both. Only `query_osm` may populate `geojson`.
