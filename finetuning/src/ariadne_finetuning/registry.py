"""Local experiment metadata.

A JSON file, not an MLOps platform. It answers one question: which adapter came
from which dataset and base model, how did it score, and has a human approved
it? Promotion into Ariadne is a manual decision recorded here, never an
automatic consequence of a good metric.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class ExperimentStatus(str, Enum):
    """Promotion state of one training run."""

    EXPERIMENTAL = "experimental"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass(frozen=True)
class ExperimentRecord:
    """Everything needed to reproduce and judge one adapter."""

    experiment_id: str
    base_model_id: str
    dataset_version: str
    dataset_digest: str
    task: str
    adapter_path: str
    training_config: dict[str, Any]
    baseline_metrics: dict[str, Any]
    finetuned_metrics: dict[str, Any] | None = None
    status: ExperimentStatus = ExperimentStatus.EXPERIMENTAL
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
    git_commit: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        payload["created_at"] = self.created_at.isoformat()
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExperimentRecord:
        data = dict(payload)
        data["status"] = ExperimentStatus(data.get("status", "experimental"))
        created = data.get("created_at")
        data["created_at"] = (
            datetime.fromisoformat(created)
            if isinstance(created, str)
            else datetime.now(tz=timezone.utc)
        )
        return cls(**data)


def current_git_commit(repo_root: Path) -> str | None:
    """Best-effort commit id; absent in an archive or a Colab copy."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


class ExperimentRegistry:
    """Append-and-update JSON registry stored under ``artifacts/``."""

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> list[ExperimentRecord]:
        if not self._path.exists():
            return []
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        experiments = payload.get("experiments", []) if isinstance(payload, dict) else []
        return [ExperimentRecord.from_dict(item) for item in experiments]

    def get(self, experiment_id: str) -> ExperimentRecord | None:
        return next(
            (item for item in self.load() if item.experiment_id == experiment_id),
            None,
        )

    def record(self, experiment: ExperimentRecord) -> ExperimentRecord:
        """Insert or replace one experiment, keeping the file sorted."""
        experiments = [
            item for item in self.load() if item.experiment_id != experiment.experiment_id
        ]
        experiments.append(experiment)
        experiments.sort(key=lambda item: item.experiment_id)
        self._write(experiments)
        return experiment

    def set_status(
        self,
        experiment_id: str,
        status: ExperimentStatus,
        *,
        notes: str = "",
    ) -> ExperimentRecord:
        """Record a human promotion decision."""
        experiments = self.load()
        for index, item in enumerate(experiments):
            if item.experiment_id == experiment_id:
                updated = ExperimentRecord(
                    **{
                        **item.to_dict(),
                        "status": status,
                        "created_at": item.created_at,
                        "notes": notes or item.notes,
                    }
                )
                experiments[index] = updated
                self._write(experiments)
                return updated
        raise KeyError(f"unknown experiment '{experiment_id}'")

    def _write(self, experiments: list[ExperimentRecord]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(
                {"experiments": [item.to_dict() for item in experiments]},
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
