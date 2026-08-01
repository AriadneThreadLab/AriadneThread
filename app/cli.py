"""Command-line entry points for OSM GeoAgent operations.

Usage:

    ./.venv/bin/python -m app.cli ingest-osm-knowledge
    ./.venv/bin/python -m app.cli index-osm-knowledge
    ./.venv/bin/python -m app.cli search-osm-knowledge --query "..."
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from app.core.config import Settings, get_settings
from app.core.errors import EmbeddingError, GeoAgentError
from app.core.logging import configure_logging
from app.db.session import Database, DatabaseConfig
from app.embeddings.cache import require_model_cached
from app.embeddings.factory import build_bge_m3_provider
from app.rag.indexing import index_osm_knowledge
from app.rag.ingest import IngestConfig, ingest_osm_knowledge
from app.rag.mediawiki import MediaWikiClient
from app.rag.retriever import SessionBoundKnowledgeRetriever


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description="OSM GeoAgent maintenance commands.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser(
        "ingest-osm-knowledge",
        help="Fetch and persist the whitelisted OSM wiki documentation corpus.",
    )
    ingest.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop after the first page failure instead of continuing.",
    )

    index = subparsers.add_parser(
        "index-osm-knowledge",
        help="Embed pending knowledge_chunks with BGE-M3 into pgvector.",
    )
    index.add_argument(
        "--force",
        action="store_true",
        help="Re-embed all chunks, including those that already have vectors.",
    )

    search = subparsers.add_parser(
        "search-osm-knowledge",
        help="Diagnostic semantic search over the indexed OSM documentation corpus.",
    )
    search.add_argument("--query", required=True, help="Natural-language search query.")
    search.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of passages to return (default: RAG_TOP_K).",
    )
    return parser


async def _run_ingest(settings: Settings, *, stop_on_error: bool) -> int:
    database = Database(DatabaseConfig(url=settings.database_url))
    client = MediaWikiClient(
        api_url=settings.osm_wiki_api_url,
        user_agent=settings.osm_wiki_user_agent,
        timeout_seconds=settings.osm_wiki_timeout_seconds,
        max_retries=settings.osm_wiki_max_retries,
    )
    try:
        summary = await ingest_osm_knowledge(
            database,
            client,
            config=IngestConfig(
                max_chars=settings.rag_chunk_max_chars,
                min_chars=settings.rag_chunk_min_chars,
                stop_on_error=stop_on_error,
            ),
        )
    finally:
        await client.aclose()
        await database.dispose()

    print(
        "OSM knowledge ingest complete: "
        f"inserted={summary.inserted} updated={summary.updated} "
        f"unchanged={summary.unchanged} failed={summary.failed}"
    )
    for outcome in summary.outcomes:
        suffix = f" error={outcome.error}" if outcome.error else ""
        print(f"  - {outcome.status}: {outcome.title} (chunks={outcome.chunk_count}){suffix}")
    return 1 if summary.failed else 0


async def _run_index(settings: Settings, *, force: bool) -> int:
    try:
        cache_root = require_model_cached(settings.bge_model_name)
    except EmbeddingError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return 2

    print(
        f"Using local embedding cache at {cache_root} "
        f"(model={settings.bge_model_name}, device={settings.bge_device})"
    )
    database = Database(DatabaseConfig(url=settings.database_url))
    embedder = build_bge_m3_provider(settings)
    try:
        result = await index_osm_knowledge(
            database,
            embedder,
            batch_size=settings.bge_batch_size,
            force=force,
        )
    except EmbeddingError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return 2
    except GeoAgentError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return 1
    finally:
        await embedder.aclose()
        await database.dispose()

    print(
        "OSM knowledge index complete: "
        f"selected={result.selected} embedded={result.embedded} "
        f"skipped={result.skipped} batches={result.batches}"
    )
    return 0


async def _run_search(settings: Settings, *, query: str, top_k: int | None) -> int:
    database = Database(DatabaseConfig(url=settings.database_url))
    embedder = build_bge_m3_provider(settings)
    retriever = SessionBoundKnowledgeRetriever(database, embedder)
    limit = top_k if top_k is not None else settings.rag_top_k
    try:
        passages = await retriever.search(query, top_k=limit)
    except (EmbeddingError, GeoAgentError) as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return 1
    finally:
        await embedder.aclose()
        await database.dispose()

    if not passages:
        print("No OSM documentation passages matched.")
        return 0

    print(f"Found {len(passages)} OSM documentation passage(s) (not live map data):")
    for index, passage in enumerate(passages, start=1):
        heading = passage.document_title
        if passage.section:
            heading = f"{heading} > {passage.section}"
        snippet = " ".join(passage.content.split())[:240]
        print(
            f"{index}. score={passage.score:.4f} [{heading}]\n"
            f"   url={passage.source_url}\n"
            f"   {snippet}"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level)

    if args.command == "ingest-osm-knowledge":
        return asyncio.run(_run_ingest(settings, stop_on_error=args.stop_on_error))
    if args.command == "index-osm-knowledge":
        return asyncio.run(_run_index(settings, force=args.force))
    if args.command == "search-osm-knowledge":
        return asyncio.run(_run_search(settings, query=args.query, top_k=args.top_k))

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
