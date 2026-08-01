"""Assemble the tool registry from concrete collaborators."""

from __future__ import annotations

from app.rag.contracts import KnowledgeRetriever
from app.tools.query_osm import QueryOsmTool
from app.tools.registry import ToolRegistry
from app.tools.search_osm_knowledge import SearchOsmKnowledgeTool


def build_tool_registry(
    *,
    knowledge_retriever: KnowledgeRetriever | None = None,
    query_osm_tool: QueryOsmTool | None = None,
    rag_top_k: int = 5,
) -> ToolRegistry:
    """Register only tools that have their dependencies available."""
    registry = ToolRegistry()
    if knowledge_retriever is not None:
        registry.register(SearchOsmKnowledgeTool(knowledge_retriever, default_top_k=rag_top_k))
    if query_osm_tool is not None:
        registry.register(query_osm_tool)
    return registry
