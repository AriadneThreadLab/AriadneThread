"""Application composition root.

Builds the long-lived dependency graph once for the FastAPI lifespan. Nothing
here connects to PostgreSQL, Ollama, Overpass or Hugging Face at import time;
BGE-M3 remains lazily loaded on first embedding use.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.active_learning.factory import build_active_learning_service
from app.active_learning.service import ActiveLearningService
from app.agent.factory import build_geo_agent
from app.agent.orchestrator import PlannerExecutorAgent
from app.analytics.factory import build_analyze_features_tool
from app.core.config import Settings
from app.db.session import Database, DatabaseConfig
from app.embeddings.bge_m3 import BgeM3EmbeddingProvider
from app.embeddings.factory import build_bge_m3_provider
from app.execution_memory.factory import build_execution_memory_service
from app.execution_memory.service import ExecutionMemoryService
from app.llm.contracts import LLMProvider
from app.llm.factory import build_llm_provider
from app.osm.factory import build_query_osm_tool
from app.places.factory import build_resolve_place_tool
from app.rag.retriever import SessionBoundKnowledgeRetriever
from app.tools.analyze_features import AnalyzeFeaturesTool
from app.tools.energy_tools import build_analyze_energy_grid_tool
from app.tools.factory import build_tool_registry
from app.tools.query_osm import QueryOsmTool
from app.tools.registry import ToolRegistry
from app.tools.resolve_place import ResolvePlaceTool
from app.tools.simbench_tools import build_simbench_query_tool


@dataclass(slots=True)
class ApplicationServices:
    """Owned runtime collaborators for one application process."""

    settings: Settings
    database: Database
    embedding_provider: BgeM3EmbeddingProvider
    query_osm_tool: QueryOsmTool
    resolve_place_tool: ResolvePlaceTool
    analyze_features_tool: AnalyzeFeaturesTool
    llm_provider: LLMProvider
    tool_registry: ToolRegistry
    geo_agent: PlannerExecutorAgent
    #: Candidate selection for human review. ``None`` disables retention.
    #: It never trains, loads or promotes a model.
    active_learning: ActiveLearningService | None = None
    #: Structured analytical memory. Independent of Active Learning.
    execution_memory: ExecutionMemoryService | None = None

    async def aclose(self) -> None:
        await self.llm_provider.aclose()
        await self.query_osm_tool.aclose()
        await self.resolve_place_tool.aclose()
        await self.embedding_provider.aclose()
        await self.database.dispose()


def build_application_services(settings: Settings) -> ApplicationServices:
    """Compose settings → infrastructure → tools → registry → orchestrator."""
    database = Database(DatabaseConfig(url=settings.database_url))
    embedding_provider = build_bge_m3_provider(settings)
    retriever = SessionBoundKnowledgeRetriever(database, embedding_provider)
    query_osm_tool = build_query_osm_tool(settings)
    resolve_place_tool = build_resolve_place_tool(settings)
    analyze_features_tool = build_analyze_features_tool()
    simbench_query_tool = build_simbench_query_tool()
    analyze_energy_grid_tool = build_analyze_energy_grid_tool()
    llm_provider = build_llm_provider(settings)
    tool_registry = build_tool_registry(
        knowledge_retriever=retriever,
        query_osm_tool=query_osm_tool,
        resolve_place_tool=resolve_place_tool,
        analyze_features_tool=analyze_features_tool,
        simbench_query_tool=simbench_query_tool,
        analyze_energy_grid_tool=analyze_energy_grid_tool,
        rag_top_k=settings.rag_top_k,
    )
    execution_memory = build_execution_memory_service(settings, database, llm_provider)
    geo_agent = build_geo_agent(llm_provider, tool_registry, settings, execution_memory)
    return ApplicationServices(
        settings=settings,
        database=database,
        embedding_provider=embedding_provider,
        query_osm_tool=query_osm_tool,
        resolve_place_tool=resolve_place_tool,
        analyze_features_tool=analyze_features_tool,
        llm_provider=llm_provider,
        tool_registry=tool_registry,
        geo_agent=geo_agent,
        active_learning=build_active_learning_service(settings, database),
        execution_memory=execution_memory,
    )
