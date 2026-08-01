"""ORM metadata for the OSM knowledge persistence layer (offline)."""

from __future__ import annotations

from typing import Any, cast

import pytest
from app.core.config import BGE_M3_EMBEDDING_DIM, Settings
from app.db.base import Base
from app.db.models import EMBEDDING_DIMENSION, KnowledgeChunk, KnowledgeDocument
from pydantic import ValidationError
from sqlalchemy import Table, UniqueConstraint
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.orm import RelationshipProperty


def _settings(**overrides: object) -> Settings:
    base: dict[str, object] = {
        "database_url": "postgresql+asyncpg://user:pass@127.0.0.1:5433/osm_geoagent",
        "_env_file": None,
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _table(model: type[Any]) -> Table:
    table = cast(Table, model.__table__)
    assert isinstance(table, Table)
    return table


def test_embedding_dimension_is_exactly_1024_everywhere():
    assert EMBEDDING_DIMENSION == 1024
    assert BGE_M3_EMBEDDING_DIM == 1024
    assert _settings().embedding_dim == EMBEDDING_DIMENSION

    embedding_column = _table(KnowledgeChunk).c.embedding
    assert cast(Any, embedding_column.type).dim == 1024
    assert embedding_column.nullable is True


def test_knowledge_document_table_has_justified_columns_only():
    columns = set(_table(KnowledgeDocument).c.keys())
    assert columns == {
        "id",
        "domain",
        "title",
        "source_url",
        "source_type",
        "license",
        "retrieved_at",
        "content_hash",
        "created_at",
        "updated_at",
    }


def test_knowledge_chunk_table_has_justified_columns_only():
    columns = set(_table(KnowledgeChunk).c.keys())
    assert columns == {
        "id",
        "document_id",
        "section",
        "content",
        "chunk_index",
        "content_hash",
        "embedding",
        "created_at",
        "updated_at",
    }


def test_document_to_chunk_relationship_is_configured():
    mapper = sa_inspect(KnowledgeDocument)
    chunks_rel = mapper.relationships["chunks"]
    assert isinstance(chunks_rel, RelationshipProperty)
    assert chunks_rel.mapper.class_ is KnowledgeChunk
    assert chunks_rel.cascade.delete_orphan
    assert "delete" in chunks_rel.cascade

    chunk_mapper = sa_inspect(KnowledgeChunk)
    document_rel = chunk_mapper.relationships["document"]
    assert document_rel.mapper.class_ is KnowledgeDocument
    assert document_rel.back_populates == "chunks"


def test_document_identity_uniqueness_supports_idempotent_ingestion():
    table = _table(KnowledgeDocument)
    uniques = [c for c in table.constraints if isinstance(c, UniqueConstraint)]
    names = {constraint.name for constraint in uniques}
    assert "uq_knowledge_documents_domain_source_url" in names

    unique = next(c for c in uniques if c.name == "uq_knowledge_documents_domain_source_url")
    assert [column.name for column in unique.columns] == ["domain", "source_url"]


def test_chunk_index_uniqueness_supports_refresh_without_duplicates():
    table = _table(KnowledgeChunk)
    uniques = [c for c in table.constraints if isinstance(c, UniqueConstraint)]
    unique = next(c for c in uniques if c.name == "uq_knowledge_chunks_document_id_chunk_index")
    assert [column.name for column in unique.columns] == ["document_id", "chunk_index"]


def test_chunk_foreign_key_cascades_on_document_delete():
    foreign_keys = list(_table(KnowledgeChunk).c.document_id.foreign_keys)
    assert len(foreign_keys) == 1
    assert foreign_keys[0].column.table.name == "knowledge_documents"
    assert foreign_keys[0].ondelete == "CASCADE"


def test_models_are_registered_on_declarative_metadata():
    assert "knowledge_documents" in Base.metadata.tables
    assert "knowledge_chunks" in Base.metadata.tables


def test_settings_still_reject_non_async_database_urls():
    with pytest.raises(ValidationError, match="asyncpg"):
        _settings(database_url="postgresql://user:pass@127.0.0.1:5433/osm_geoagent")
