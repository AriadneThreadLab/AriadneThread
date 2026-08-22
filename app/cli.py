"""Command-line entry points for OSM GeoAgent operations.

Usage:

    ./.venv/bin/python -m app.cli ingest-osm-knowledge
    ./.venv/bin/python -m app.cli index-osm-knowledge
    ./.venv/bin/python -m app.cli search-osm-knowledge --query "..."
    ./.venv/bin/python -m app.cli agent-query --message "Find public parks in Berlin"
    ./.venv/bin/python -m app.cli list-active-learning-candidates --review-queue
    ./.venv/bin/python -m app.cli export-training-dataset --dataset-version v1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from app.active_learning import cli_commands as al_commands
from app.active_learning.contracts import ExportTask, ReviewStatus
from app.agent.contracts import GeoAgentRequest
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

    agent = subparsers.add_parser(
        "agent-query",
        help="Developer diagnostic: run the bounded planner-executor agent once.",
    )
    agent.add_argument(
        "--message",
        required=True,
        help="Natural-language GIS request to send to the agent.",
    )

    subparsers.add_parser(
        "diagnose-ollama",
        help="Developer diagnostic: probe Ollama chat, capabilities, and tool modes.",
    )
    subparsers.add_parser(
        "diagnose-avalai",
        help="Developer diagnostic: probe AvalAI chat and native tool calling.",
    )

    _add_active_learning_parsers(subparsers)
    return parser


def _add_active_learning_parsers(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Review-queue and dataset-export commands (no training happens here)."""
    listing = subparsers.add_parser(
        "list-active-learning-candidates",
        help="List retained candidates, highest informativeness first.",
    )
    listing.add_argument(
        "--status",
        choices=[status.value for status in ReviewStatus],
        default=None,
    )
    listing.add_argument(
        "--review-queue",
        action="store_true",
        help="Only candidates at or above the human-review score threshold.",
    )
    listing.add_argument("--limit", type=int, default=20)
    listing.add_argument("--offset", type=int, default=0)

    show = subparsers.add_parser(
        "show-active-learning-candidate",
        help="Print one candidate as JSON.",
    )
    show.add_argument("--candidate-id", required=True)

    approve = subparsers.add_parser(
        "approve-active-learning-candidate",
        help="Approve a validated successful candidate for training use.",
    )
    approve.add_argument("--candidate-id", required=True)
    approve.add_argument("--reviewer", default=None)
    approve.add_argument("--note", default=None)

    reject = subparsers.add_parser(
        "reject-active-learning-candidate",
        help="Reject a candidate so it can never be exported.",
    )
    reject.add_argument("--candidate-id", required=True)
    reject.add_argument("--reviewer", default=None)
    reject.add_argument("--note", default=None)

    correct = subparsers.add_parser(
        "correct-active-learning-candidate",
        help="Attach a human-corrected target plan and approve it.",
    )
    correct.add_argument("--candidate-id", required=True)
    correct.add_argument(
        "--corrected-output-file",
        required=True,
        help="JSON file holding the corrected plan for the chosen task.",
    )
    correct.add_argument(
        "--task",
        choices=[task.value for task in ExportTask],
        default=ExportTask.ANALYSIS_PLAN.value,
    )
    correct.add_argument("--reviewer", default=None)
    correct.add_argument("--note", default=None)

    subparsers.add_parser(
        "active-learning-stats",
        help="Print review-queue counters as JSON.",
    )

    export = subparsers.add_parser(
        "export-training-dataset",
        help="Write a versioned SFT dataset from approved candidates.",
    )
    export.add_argument("--dataset-version", required=True, help="For example: v1.")
    export.add_argument(
        "--task",
        choices=[task.value for task in ExportTask],
        default=ExportTask.ANALYSIS_PLAN.value,
    )
    export.add_argument(
        "--output-dir",
        default="finetuning/data",
        help="Directory for the split files, manifest and target JSON Schema.",
    )
    export.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be exported without writing or freezing anything.",
    )


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


async def _run_diagnose_ollama(settings: Settings) -> int:
    from app.llm.diagnostics import run_ollama_diagnostics

    report = await run_ollama_diagnostics(settings)
    print(report.format_text())
    return 0 if report.basic_chat.startswith("PASS") else 1


async def _run_diagnose_avalai(settings: Settings) -> int:
    from app.llm.diagnostics import run_avalai_diagnostics

    report = await run_avalai_diagnostics(settings)
    print(report.format_text())
    return 0 if report.basic_chat.startswith("PASS") else 1


async def _run_agent_query(settings: Settings, *, message: str) -> int:
    """Real configured dependencies; for developer diagnosis only."""
    from app.bootstrap import build_application_services

    services = build_application_services(settings)
    try:
        result = await services.geo_agent.run(GeoAgentRequest(message=message))
    except GeoAgentError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return 1
    finally:
        await services.aclose()

    print(f"stop_reason={result.stop_reason} model={result.model}")
    print(f"answer:\n{result.answer}\n")
    if result.feature_count is not None:
        print(f"feature_count={result.feature_count}")
    if result.effective_limit is not None:
        print(f"effective_limit={result.effective_limit}")
    if result.scope_summary:
        print(f"scope_summary={result.scope_summary}")
    if result.validated_tags:
        print(f"validated_tags={result.validated_tags}")
    if result.analysis is not None:
        analysis = result.analysis
        primary = analysis.decision_trace.final_primary_metric
        print(
            f"analysis status={analysis.status} "
            f"type={analysis.plan.analysis_type} primary_metric={primary}"
        )
        if analysis.comparison is not None:
            print(f"analysis.comparison={analysis.comparison.model_dump()}")
        if analysis.result is not None:
            for target in analysis.result.targets:
                print(
                    f"  target id={target.target_id} label={target.label} "
                    f"dataset_ref={target.dataset_ref}"
                )
                for metric in target.metrics:
                    print(
                        f"    metric={metric.metric} role={metric.role} "
                        f"value={metric.value} notes={list(metric.notes)}"
                    )
    if result.overpass_query:
        print(f"overpass_query:\n{result.overpass_query}\n")
    if result.warnings:
        print("warnings:")
        for warning in result.warnings:
            print(f"  - {warning}")
    if result.errors:
        print("errors:")
        for error in result.errors:
            print(f"  - {error}")
    if result.sources:
        print("sources:")
        for source in result.sources:
            print(
                f"  - [{source.kind}] {source.title}" + (f" <{source.url}>" if source.url else "")
            )
    print("trace:")
    for event in result.trace:
        tool = f" tool={event.tool_name}" if event.tool_name else ""
        code = f" code={event.error_code}" if event.error_code else ""
        print(f"  - {event.kind} r{event.round_index}:{tool}{code} {event.message}")
        if event.details:
            safe = {
                key: event.details[key]
                for key in (
                    "status",
                    "duration_ms",
                    "estimated_tokens",
                    "system_prompt_chars",
                    "total_message_chars",
                    "eligible_tool_count",
                    "place_ref",
                    "dataset_ref",
                    "label",
                    "feature_count",
                    "truncated",
                    "effective_limit",
                )
                if key in event.details
            }
            if safe:
                print(f"    details={safe}")
    if result.geojson is not None:
        # Compact diagnostic: type + feature count only (not a full dump).
        features = result.geojson.get("features")
        count = len(features) if isinstance(features, list) else 0
        print(f"geojson: FeatureCollection features={count}")
        print(json.dumps({"type": result.geojson.get("type"), "feature_count": count}))
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
    if args.command == "agent-query":
        return asyncio.run(_run_agent_query(settings, message=args.message))
    if args.command == "diagnose-ollama":
        return asyncio.run(_run_diagnose_ollama(settings))
    if args.command == "diagnose-avalai":
        return asyncio.run(_run_diagnose_avalai(settings))

    active_learning_exit = _dispatch_active_learning(settings, args)
    if active_learning_exit is not None:
        return active_learning_exit

    parser.error(f"unknown command: {args.command}")
    return 2


def _dispatch_active_learning(settings: Settings, args: argparse.Namespace) -> int | None:
    """Run an active-learning command, or return ``None`` if none matched."""
    command = args.command
    if command == "list-active-learning-candidates":
        return asyncio.run(
            al_commands.run_list(
                settings,
                status=args.status,
                review_queue=args.review_queue,
                limit=args.limit,
                offset=args.offset,
            )
        )
    if command == "show-active-learning-candidate":
        return asyncio.run(al_commands.run_show(settings, candidate_id=args.candidate_id))
    if command in {"approve-active-learning-candidate", "reject-active-learning-candidate"}:
        status = (
            ReviewStatus.APPROVED
            if command == "approve-active-learning-candidate"
            else ReviewStatus.REJECTED
        )
        return asyncio.run(
            al_commands.run_review(
                settings,
                candidate_id=args.candidate_id,
                status=status,
                reviewer=args.reviewer,
                note=args.note,
            )
        )
    if command == "correct-active-learning-candidate":
        return asyncio.run(
            al_commands.run_correct(
                settings,
                candidate_id=args.candidate_id,
                corrected_output_file=Path(args.corrected_output_file),
                task=ExportTask(args.task),
                reviewer=args.reviewer,
                note=args.note,
            )
        )
    if command == "active-learning-stats":
        return asyncio.run(al_commands.run_stats(settings))
    if command == "export-training-dataset":
        return asyncio.run(
            al_commands.run_export(
                settings,
                dataset_version=args.dataset_version,
                task=ExportTask(args.task),
                output_dir=Path(args.output_dir),
                dry_run=args.dry_run,
            )
        )
    return None


if __name__ == "__main__":
    sys.exit(main())
