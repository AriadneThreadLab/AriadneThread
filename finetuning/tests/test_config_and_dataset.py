"""Configuration loading and dataset validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ariadne_finetuning.config import (
    ConfigError,
    apply_env_overrides,
    config_from_mapping,
    load_config,
)
from ariadne_finetuning.dataset import (
    DatasetError,
    load_bundle,
    load_jsonl,
    validate_bundle,
)
from conftest import plan, sample_line

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "finetuning" / "configs" / "qlora_analysis_plan.yaml"


def test_shipped_config_loads_with_conservative_qlora_defaults():
    config = load_config(CONFIG_PATH)

    assert config.task == "analysis_plan"
    assert config.quantization.load_in_4bit is True
    assert config.quantization.quant_type == "nf4"
    assert config.quantization.double_quant is True
    assert config.training.train_batch_size == 1
    assert config.training.gradient_accumulation_steps > 1
    assert config.training.gradient_checkpointing is True
    assert config.lora.target_modules == ("q_proj", "k_proj", "v_proj", "o_proj")


def test_base_model_is_configurable_and_not_hard_coded():
    config = load_config(CONFIG_PATH)
    overridden = apply_env_overrides(
        config,
        {"BASE_MODEL_ID": "org/other-instruct", "LORA_R": "8", "MAX_SEQ_LENGTH": "512"},
    )

    assert overridden.base_model_id == "org/other-instruct"
    assert overridden.lora.r == 8
    assert overridden.training.max_seq_length == 512
    # The original object is untouched.
    assert config.lora.r == 16


def test_bad_override_is_reported_clearly():
    config = load_config(CONFIG_PATH)
    with pytest.raises(ConfigError):
        apply_env_overrides(config, {"LORA_R": "not-a-number"})


def test_missing_required_keys_are_rejected():
    with pytest.raises(ConfigError):
        config_from_mapping({"experiment_id": "x"})


def test_unknown_section_keys_are_rejected():
    with pytest.raises(ConfigError):
        config_from_mapping(
            {
                "experiment_id": "x",
                "base_model_id": "y",
                "dataset_dir": "data",
                "dataset_version": "v1",
                "lora": {"rank": 8},
            }
        )


def test_bundle_loads_splits_manifest_and_authoritative_schema(dataset_dir):
    bundle = load_bundle(dataset_dir, task="analysis_plan", dataset_version="v1")

    assert (len(bundle.train), len(bundle.validation), len(bundle.test)) == (2, 1, 1)
    assert bundle.manifest["target_schema_name"] == "AnalysisPlan"
    assert bundle.target_schema["title"] == "AnalysisPlan"
    assert validate_bundle(bundle) == []


def test_test_digest_is_stable_and_sensitive(dataset_dir):
    bundle = load_bundle(dataset_dir, task="analysis_plan", dataset_version="v1")
    first = bundle.test_digest

    reloaded = load_bundle(dataset_dir, task="analysis_plan", dataset_version="v1")
    assert reloaded.test_digest == first

    path = dataset_dir / "analysis_plan_v1.test.jsonl"
    path.write_text(
        sample_line("A different evaluation question entirely?", plan()) + "\n",
        encoding="utf-8",
    )
    changed = load_bundle(dataset_dir, task="analysis_plan", dataset_version="v1")
    assert changed.test_digest != first


def test_leakage_between_train_and_test_is_reported(dataset_dir):
    leaked = (dataset_dir / "analysis_plan_v1.train.jsonl").read_text(encoding="utf-8")
    (dataset_dir / "analysis_plan_v1.test.jsonl").write_text(
        leaked.splitlines(keepends=True)[0], encoding="utf-8"
    )

    bundle = load_bundle(dataset_dir, task="analysis_plan", dataset_version="v1")
    problems = validate_bundle(bundle)

    assert any("duplicates a training prompt" in item for item in problems)


def test_empty_test_split_is_a_validation_failure(dataset_dir):
    (dataset_dir / "analysis_plan_v1.test.jsonl").write_text("", encoding="utf-8")

    bundle = load_bundle(dataset_dir, task="analysis_plan", dataset_version="v1")

    assert any("test split is empty" in item for item in validate_bundle(bundle))


def test_malformed_jsonl_is_rejected(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text('{"messages": [{"role": "user", "content": "only one"}]}\n', encoding="utf-8")

    with pytest.raises(DatasetError):
        load_jsonl(path)


def test_manifest_version_mismatch_is_rejected(dataset_dir):
    manifest_path = dataset_dir / "analysis_plan_v1.manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["dataset_version"] = "v9"
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DatasetError):
        load_bundle(dataset_dir, task="analysis_plan", dataset_version="v1")
