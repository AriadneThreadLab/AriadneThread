"""Offline checks for the knowledge-table Alembic revision."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory
from app.db.models import EMBEDDING_DIMENSION

REPO_ROOT = Path(__file__).resolve().parents[1]
MIGRATION_PATH = REPO_ROOT / "alembic" / "versions" / "20260801_0001_knowledge_tables.py"


def _load_migration_module():
    spec = importlib.util.spec_from_file_location(
        "knowledge_tables_migration",
        MIGRATION_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_alembic_history_is_linear_and_starts_at_knowledge_revision():
    config = Config(str(REPO_ROOT / "alembic.ini"))
    script = ScriptDirectory.from_config(config)
    # One head keeps `alembic upgrade head` unambiguous.
    assert len(script.get_heads()) == 1
    revisions = [revision.revision for revision in script.walk_revisions()]
    assert revisions[-1] == "20260801_0001"


def test_knowledge_migration_module_imports_and_declares_revision():
    module = _load_migration_module()
    assert module.revision == "20260801_0001"
    assert module.down_revision is None
    assert module.EMBEDDING_DIMENSION == EMBEDDING_DIMENSION == 1024
    assert callable(module.upgrade)
    assert callable(module.downgrade)


def test_knowledge_migration_source_enables_required_extensions():
    source = MIGRATION_PATH.read_text(encoding="utf-8")
    assert "CREATE EXTENSION IF NOT EXISTS postgis" in source
    assert "CREATE EXTENSION IF NOT EXISTS vector" in source
    assert "VECTOR(EMBEDDING_DIMENSION)" in source
    assert "knowledge_documents" in source
    assert "knowledge_chunks" in source
    assert "uq_knowledge_documents_domain_source_url" in source
    assert "uq_knowledge_chunks_document_id_chunk_index" in source
    # Downgrade must not drop extensions (other objects may depend on them).
    downgrade_body = source.split("def downgrade")[1]
    assert "DROP EXTENSION" not in downgrade_body


def test_alembic_env_imports_knowledge_models():
    env_source = (REPO_ROOT / "alembic" / "env.py").read_text(encoding="utf-8")
    assert "from app.db.models import" in env_source
    assert "KnowledgeDocument" in env_source
    assert "KnowledgeChunk" in env_source
