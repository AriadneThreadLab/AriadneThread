"""Shared fixtures for the fine-tuning package.

No model is downloaded, no GPU is required and no Ariadne database is touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLD_DIR = REPO_ROOT / "finetuning" / "evaluation" / "gold"

ANALYSIS_PLAN_SCHEMA: dict[str, object] = json.loads(
    (GOLD_DIR / "analysis_plan_gold_v1.schema.json").read_text(encoding="utf-8")
)


def plan(
    *,
    analysis_type: str = "comparison",
    feature_concept: str = "public parks",
    goal: str = "which campus area has more parks",
    targets: tuple[tuple[str, str, str], ...] = (
        ("ut", "University of Tehran", "osm_result_1"),
        ("sharif", "Sharif University of Technology", "osm_result_2"),
    ),
    metric: str = "count",
) -> dict[str, object]:
    return {
        "analysis_type": analysis_type,
        "feature_concept": feature_concept,
        "comparison_goal": goal,
        "targets": [
            {
                "target_id": target_id,
                "label": label,
                "dataset_ref": dataset_ref,
                "reference_points": [],
            }
            for target_id, label, dataset_ref in targets
        ],
        "metrics": [
            {
                "metric": metric,
                "role": "primary",
                "inferred_goal": "abundance",
                "claimed_rule_ids": ["ABUNDANCE_COUNT_001"],
                "user_explicit": False,
                "property_key": None,
                "direction": "higher_is_better",
                "ratio": None,
            }
        ],
    }


def sample_line(prompt: str, target: dict[str, object]) -> str:
    return json.dumps(
        {
            "messages": [
                {"role": "user", "content": prompt},
                {
                    "role": "assistant",
                    "content": json.dumps(
                        target, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    ),
                },
            ]
        },
        ensure_ascii=False,
        sort_keys=True,
    )


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    """A minimal but complete exported dataset (three splits + metadata)."""
    directory = tmp_path / "data"
    directory.mkdir()
    stem = "analysis_plan_v1"

    splits = {
        "train": [
            ("Compare public parks within 2 km of Campus A and Campus B.", plan()),
            (
                "Compare libraries within 1 km of Station A and Station B.",
                plan(feature_concept="libraries", goal="which station has more libraries"),
            ),
        ],
        "validation": [
            (
                "Compare clinics within 3 km of District A and District B.",
                plan(feature_concept="clinics", goal="which district has more clinics"),
            )
        ],
        "test": [
            (
                "Compare museums within 4 km of Quarter A and Quarter B.",
                plan(feature_concept="museums", goal="which quarter has more museums"),
            )
        ],
    }
    for split, rows in splits.items():
        (directory / f"{stem}.{split}.jsonl").write_text(
            "".join(sample_line(prompt, target) + "\n" for prompt, target in rows),
            encoding="utf-8",
        )

    (directory / f"{stem}.manifest.json").write_text(
        json.dumps(
            {
                "dataset_version": "v1",
                "task": "analysis_plan",
                "target_schema_name": "AnalysisPlan",
                "example_count": 4,
                "split_counts": {"train": 2, "validation": 1, "test": 1},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (directory / f"{stem}.schema.json").write_text(
        json.dumps(ANALYSIS_PLAN_SCHEMA, indent=2, sort_keys=True), encoding="utf-8"
    )
    return directory
