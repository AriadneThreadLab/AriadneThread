"""Pipeline orchestration, experiment registry and the no-auto-deploy rule."""

from __future__ import annotations

import ast
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from ariadne_finetuning.config import config_from_mapping
from ariadne_finetuning.dataset import DatasetBundle, load_bundle
from ariadne_finetuning.evaluation import ModelRunner
from ariadne_finetuning.modeling import TRAINING_REQUIREMENTS
from ariadne_finetuning.pipeline import PipelineError, run_pipeline
from ariadne_finetuning.registry import (
    ExperimentRecord,
    ExperimentRegistry,
    ExperimentStatus,
)

PACKAGE_DIR = Path(__file__).resolve().parents[1] / "src" / "ariadne_finetuning"
NOW = datetime(2026, 8, 16, 12, 0, tzinfo=timezone.utc)


class ScriptedRunner:
    def __init__(self, reply: str, label: str) -> None:
        self._reply = reply
        self._label = label
        self.seen: list[str] = []

    @property
    def label(self) -> str:
        return self._label

    def generate(self, prompt: str) -> str:
        self.seen.append(prompt)
        return self._reply


def build_config(dataset_dir: Path, output_dir: Path):
    return config_from_mapping(
        {
            "experiment_id": "exp-001",
            "base_model_id": "org/tiny-instruct",
            "dataset_dir": str(dataset_dir),
            "dataset_version": "v1",
            "task": "analysis_plan",
            "output_dir": str(output_dir),
        }
    )


def test_pipeline_evaluates_baseline_trains_then_re_evaluates(dataset_dir, tmp_path):
    config = build_config(dataset_dir, tmp_path / "artifacts")
    bundle = load_bundle(dataset_dir, task="analysis_plan", dataset_version="v1")
    perfect = bundle.test[0].completion
    runners: dict[str, ScriptedRunner] = {}
    trained: list[tuple[str, int]] = []

    def runner_factory(_config, adapter: Path | None) -> ModelRunner:
        label = "finetuned" if adapter else "base"
        runner = ScriptedRunner("no plan here" if adapter is None else perfect, label)
        runners[label] = runner
        return runner

    def trainer(_config, training_bundle: DatasetBundle) -> Path:
        trained.append((training_bundle.dataset_version, len(training_bundle.train)))
        adapter = tmp_path / "artifacts" / "exp-001" / "adapter"
        adapter.mkdir(parents=True, exist_ok=True)
        return adapter

    registry = ExperimentRegistry(tmp_path / "artifacts" / "registry.json")
    result = run_pipeline(
        config,
        runner_factory=runner_factory,
        trainer=trainer,
        registry=registry,
        now=NOW,
    )

    assert trained == [("v1", 2)], "training saw the train split, not the test split"
    assert runners["base"].seen == runners["finetuned"].seen
    assert result.baseline.metrics.valid_analysis_plan_rate == 0.0
    assert result.finetuned is not None
    assert result.finetuned.metrics.valid_analysis_plan_rate == 1.0
    assert result.comparison is not None
    assert result.comparison["metrics"]["valid_analysis_plan_rate"]["delta"] == 1.0


def test_pipeline_records_an_experimental_status_and_never_promotes(dataset_dir, tmp_path):
    config = build_config(dataset_dir, tmp_path / "artifacts")
    registry = ExperimentRegistry(tmp_path / "artifacts" / "registry.json")

    result = run_pipeline(
        config,
        runner_factory=lambda _config, adapter: ScriptedRunner(
            "{}", "finetuned" if adapter else "base"
        ),
        trainer=lambda _config, _bundle: tmp_path / "adapter",
        registry=registry,
        now=NOW,
    )

    assert result.experiment is not None
    assert result.experiment.status is ExperimentStatus.EXPERIMENTAL
    stored = registry.get("exp-001")
    assert stored is not None
    assert stored.status is ExperimentStatus.EXPERIMENTAL
    assert stored.dataset_version == "v1"
    assert stored.baseline_metrics and stored.finetuned_metrics


def test_baseline_only_run_stops_before_training(dataset_dir, tmp_path):
    config = build_config(dataset_dir, tmp_path / "artifacts")

    def trainer(_config, _bundle):
        raise AssertionError("training must not run for a baseline-only evaluation")

    result = run_pipeline(
        config,
        runner_factory=lambda _config, _adapter: ScriptedRunner("{}", "base"),
        trainer=trainer,
        registry=ExperimentRegistry(tmp_path / "registry.json"),
        baseline_only=True,
        now=NOW,
    )

    assert result.finetuned is None
    assert result.adapter_path is None
    assert result.experiment is None


def test_pipeline_refuses_a_leaking_dataset(dataset_dir, tmp_path):
    leaked = (dataset_dir / "analysis_plan_v1.train.jsonl").read_text(encoding="utf-8")
    (dataset_dir / "analysis_plan_v1.test.jsonl").write_text(
        leaked.splitlines(keepends=True)[0], encoding="utf-8"
    )
    config = build_config(dataset_dir, tmp_path / "artifacts")

    with pytest.raises(PipelineError):
        run_pipeline(
            config,
            runner_factory=lambda _config, _adapter: ScriptedRunner("{}", "base"),
            trainer=lambda _config, _bundle: tmp_path / "adapter",
            registry=ExperimentRegistry(tmp_path / "registry.json"),
            now=NOW,
        )


def test_promotion_is_an_explicit_human_decision(tmp_path):
    registry = ExperimentRegistry(tmp_path / "registry.json")
    registry.record(
        ExperimentRecord(
            experiment_id="exp-001",
            base_model_id="org/tiny-instruct",
            dataset_version="v1",
            dataset_digest="digest",
            task="analysis_plan",
            adapter_path="artifacts/exp-001/adapter",
            training_config={},
            baseline_metrics={"valid_analysis_plan_rate": 0.1},
            finetuned_metrics={"valid_analysis_plan_rate": 0.8},
            created_at=NOW,
        )
    )

    promoted = registry.set_status("exp-001", ExperimentStatus.APPROVED, notes="reviewed")

    assert promoted.status is ExperimentStatus.APPROVED
    assert promoted.notes == "reviewed"
    assert registry.get("exp-001").status is ExperimentStatus.APPROVED
    with pytest.raises(KeyError):
        registry.set_status("missing", ExperimentStatus.APPROVED)


def test_registry_file_is_readable_json(tmp_path):
    registry = ExperimentRegistry(tmp_path / "registry.json")
    registry.record(
        ExperimentRecord(
            experiment_id="exp-001",
            base_model_id="org/tiny-instruct",
            dataset_version="v1",
            dataset_digest="digest",
            task="analysis_plan",
            adapter_path="artifacts/exp-001/adapter",
            training_config={"lora": {"r": 16}},
            baseline_metrics={},
            created_at=NOW,
        )
    )

    payload = json.loads((tmp_path / "registry.json").read_text(encoding="utf-8"))
    assert payload["experiments"][0]["experiment_id"] == "exp-001"
    assert payload["experiments"][0]["status"] == "experimental"


# --- boundaries -----------------------------------------------------------


def test_importing_the_package_loads_no_training_stack():
    for module in ("ariadne_finetuning.pipeline", "ariadne_finetuning.training"):
        __import__(module)
    loaded = set(TRAINING_REQUIREMENTS) & set(sys.modules)
    assert loaded == set(), f"heavy packages imported at module scope: {sorted(loaded)}"


def test_training_imports_stay_inside_functions():
    """Module-level ML imports would break offline validation."""
    offenders: dict[str, set[str]] = {}
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        top_level: set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                top_level.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                top_level.add(node.module.split(".")[0])
        forbidden = top_level & set(TRAINING_REQUIREMENTS)
        if forbidden:
            offenders[path.name] = forbidden
    assert offenders == {}


def test_finetuning_package_never_imports_the_ariadne_application():
    offenders: dict[str, set[str]] = {}
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                modules.add(node.module.split(".")[0])
        forbidden = modules & {"app", "sqlalchemy", "alembic", "fastapi"}
        if forbidden:
            offenders[path.name] = forbidden
    assert offenders == {}, "training code must read exported files, not the application database"


def test_missing_dependencies_produce_an_actionable_error():
    """Training deps are absent in the offline environment; say so usefully."""
    from ariadne_finetuning.modeling import (
        MissingDependencyError,
        missing_requirements,
        require_dependencies,
    )

    if not missing_requirements():
        pytest.skip("the training stack is installed in this environment")

    with pytest.raises(MissingDependencyError) as excinfo:
        require_dependencies()
    assert "pip install" in str(excinfo.value)
