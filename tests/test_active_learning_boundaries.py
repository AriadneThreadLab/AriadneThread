"""Architectural boundaries between the agent, active learning and training.

Active learning must never degrade a response, and nothing in the running
application may reach into the fine-tuning pipeline.
"""

from __future__ import annotations

import ast
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pytest
from app.active_learning.contracts import ActiveLearningCandidate, ReviewStatus
from app.active_learning.repository import InMemoryCandidateRepository
from app.active_learning.service import ActiveLearningService, SelectionPolicy
from app.agent.contracts import GeoAgentRequest, GeoAgentResponse, TraceEvent
from app.bootstrap import ApplicationServices
from app.core.config import Settings
from app.main import create_app

from tests.active_learning_fixtures import COMPARISON_QUERY, successful_run

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / "app"
FINETUNING_DIR = REPO_ROOT / "finetuning"

#: Training-stack modules that must never be importable from the web service.
TRAINING_PACKAGES = frozenset(
    {"torch", "transformers", "peft", "trl", "bitsandbytes", "accelerate", "datasets"}
)


class FakeAgent:
    def __init__(self, result: GeoAgentResponse) -> None:
        self._result = result

    async def run(self, request: GeoAgentRequest) -> GeoAgentResponse:
        return self._result


class ExplodingRepository(InMemoryCandidateRepository):
    """Every persistence call fails, as a broken database would."""

    async def add(self, candidate: ActiveLearningCandidate) -> ActiveLearningCandidate:
        raise RuntimeError("active learning storage is unavailable")

    async def find_by_query_hash(self, query_hash: str) -> list[ActiveLearningCandidate]:
        raise RuntimeError("active learning storage is unavailable")


@pytest.fixture
def api_settings() -> Settings:
    return Settings(
        app_env="test",
        database_url="postgresql+asyncpg://user:pass@127.0.0.1:5433/osm_geoagent_test",
        ollama_model="deepseek-r1:7b",
        agent_request_timeout_seconds=5.0,
        _env_file=None,  # type: ignore[call-arg]
    )


def build_services(
    settings: Settings,
    agent: FakeAgent,
    active_learning: ActiveLearningService | None,
) -> ApplicationServices:
    registry = MagicMock()
    registry.names = ("query_osm",)
    return ApplicationServices(
        settings=settings,
        database=MagicMock(),
        embedding_provider=MagicMock(is_loaded=False),
        query_osm_tool=MagicMock(),
        resolve_place_tool=MagicMock(),
        analyze_features_tool=MagicMock(),
        llm_provider=MagicMock(),
        tool_registry=registry,
        geo_agent=agent,  # type: ignore[arg-type]
        active_learning=active_learning,
    )


async def call_agent(app, message: str = COMPARISON_QUERY) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as http,
    ):
        return await http.post("/api/v1/agent/query", json={"message": message})


async def test_active_learning_failure_never_blocks_the_agent_response(api_settings):
    service = ActiveLearningService(ExplodingRepository(), policy=SelectionPolicy())
    app = create_app(
        api_settings,
        services=build_services(api_settings, FakeAgent(successful_run()), service),
    )

    response = await call_agent(app)

    assert response.status_code == 200
    assert response.json()["answer"].startswith("Found 12 parks")


async def test_successful_run_reaches_the_review_queue_through_the_api(api_settings):
    repository = InMemoryCandidateRepository()
    service = ActiveLearningService(repository, policy=SelectionPolicy())
    app = create_app(
        api_settings,
        services=build_services(api_settings, FakeAgent(successful_run()), service),
    )

    response = await call_agent(app)

    assert response.status_code == 200
    stored = await repository.list_candidates(limit=5)
    assert len(stored) == 1
    assert stored[0].request_id == response.json()["request_id"]
    assert stored[0].review_status is ReviewStatus.PENDING


async def test_feedback_endpoint_queues_a_candidate_without_approving_it(api_settings):
    repository = InMemoryCandidateRepository()
    service = ActiveLearningService(repository, policy=SelectionPolicy())
    app = create_app(
        api_settings,
        services=build_services(api_settings, FakeAgent(successful_run()), service),
    )

    transport = httpx.ASGITransport(app=app)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=transport, base_url="http://test") as http,
    ):
        query = await http.post("/api/v1/agent/query", json={"message": COMPARISON_QUERY})
        request_id = query.json()["request_id"]
        feedback = await http.post(
            "/api/v1/agent/feedback",
            json={
                "request_id": request_id,
                "sentiment": "negative",
                "failure_category": "wrong_metric",
                "note": "should have used density",
            },
        )

    assert feedback.status_code == 200
    body = feedback.json()
    assert body["accepted"] is True
    assert body["review_status"] == "pending"
    candidate = await repository.get_by_request(request_id)
    assert candidate is not None
    assert candidate.approved_for_training is False


async def test_active_learning_can_be_disabled_entirely(api_settings):
    app = create_app(
        api_settings,
        services=build_services(api_settings, FakeAgent(successful_run()), None),
    )

    response = await call_agent(app)

    assert response.status_code == 200


async def test_disabled_by_settings_produces_no_service():
    from app.active_learning.factory import build_active_learning_service

    settings = Settings(
        app_env="test",
        database_url="postgresql+asyncpg://user:pass@127.0.0.1:5433/osm_geoagent_test",
        active_learning_enabled=False,
        _env_file=None,  # type: ignore[call-arg]
    )
    assert build_active_learning_service(settings, MagicMock()) is None


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module.split(".")[0])
    return modules


def test_active_learning_never_imports_a_training_stack():
    offenders: dict[str, set[str]] = {}
    for path in sorted((APP_DIR / "active_learning").rglob("*.py")):
        forbidden = _imported_modules(path) & (TRAINING_PACKAGES | {"finetuning"})
        if forbidden:
            offenders[path.name] = forbidden
    assert offenders == {}


def test_application_code_never_imports_the_finetuning_pipeline():
    offenders: dict[str, set[str]] = {}
    for path in sorted(APP_DIR.rglob("*.py")):
        forbidden = _imported_modules(path) & (
            TRAINING_PACKAGES | {"finetuning", "ariadne_finetuning"}
        )
        if forbidden:
            offenders[str(path.relative_to(REPO_ROOT))] = forbidden
    assert offenders == {}


def test_active_learning_exposes_no_training_entry_point():
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((APP_DIR / "active_learning").rglob("*.py"))
    )
    for forbidden in ("SFTTrainer", "LoraConfig", "from_pretrained", "adapter_model"):
        assert forbidden not in source


async def test_fastapi_startup_does_not_load_the_training_stack(api_settings):
    import sys

    app = create_app(
        api_settings,
        services=build_services(api_settings, FakeAgent(successful_run()), None),
    )
    async with app.router.lifespan_context(app):
        loaded = TRAINING_PACKAGES & set(sys.modules)
    assert loaded == set(), f"training packages loaded during startup: {sorted(loaded)}"


def test_training_artifacts_are_gitignored():
    patterns = {
        line.strip()
        for line in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }
    assert "finetuning/artifacts/" in patterns
    assert "finetuning/data/*.jsonl" in patterns


def test_trace_event_never_carries_reasoning_into_a_candidate():
    """Trace details are operational metadata only."""
    event = TraceEvent(
        kind="llm_turn",
        message="comparison plan accepted",
        details={"status": "completed", "radius_meters": 2000},
    )
    assert "thinking" not in (event.details or {})
    assert event.details is not None and set(event.details) <= {
        "status",
        "radius_meters",
        "target_count",
        "feature_concept",
        "comparison_plan",
    }
