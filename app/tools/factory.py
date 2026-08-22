"""Assemble the tool registry from concrete collaborators."""

from __future__ import annotations

from app.rag.contracts import KnowledgeRetriever
from app.tools.analyze_features import AnalyzeFeaturesTool
from app.tools.query_osm import QueryOsmTool
from app.tools.registry import ToolRegistry
from app.tools.resolve_place import ResolvePlaceTool
from app.tools.search_osm_knowledge import SearchOsmKnowledgeTool


def build_tool_registry(
    *,
    knowledge_retriever: KnowledgeRetriever | None = None,
    query_osm_tool: QueryOsmTool | None = None,
    analyze_features_tool: AnalyzeFeaturesTool | None = None,
    resolve_place_tool: ResolvePlaceTool | None = None,
    rag_top_k: int = 5,
) -> ToolRegistry:
    """Register only tools that have their dependencies available."""
    registry = ToolRegistry()
    if knowledge_retriever is not None:
        registry.register(SearchOsmKnowledgeTool(knowledge_retriever, default_top_k=rag_top_k))
    if resolve_place_tool is not None:
        registry.register(resolve_place_tool)
    if query_osm_tool is not None:
        registry.register(query_osm_tool)
    if analyze_features_tool is not None:
        registry.register(analyze_features_tool)
    return registry
