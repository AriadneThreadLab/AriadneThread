"""Fine-tuning configuration.

Plain dataclasses plus a YAML/JSON loader and environment overrides. Importing
this module pulls in no machine-learning dependency, which is what lets the
offline test suite exercise the pipeline without a GPU or a model download.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

#: Attention projections are the conventional, memory-cheap QLoRA target set.
DEFAULT_LORA_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
)


class ConfigError(ValueError):
    """The configuration file or environment override is unusable."""


@dataclass(frozen=True)
class LoraSettings:
    """LoRA adapter shape."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = DEFAULT_LORA_TARGET_MODULES
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass(frozen=True)
class QuantizationSettings:
    """4-bit base-model loading (QLoRA)."""

    load_in_4bit: bool = True
    quant_type: str = "nf4"
    double_quant: bool = True
    compute_dtype: str = "bfloat16"


@dataclass(frozen=True)
class TrainingSettings:
    """Conservative defaults sized for a single Colab-class GPU.

    These are starting points, not a hardware compatibility claim: whether a
    given base model fits depends on the model, the sequence length and the
    card. Measure before trusting them.
    """

    learning_rate: float = 2e-4
    num_epochs: float = 3.0
    train_batch_size: int = 1
    eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 1024
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    logging_steps: int = 10
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_8bit"
    lr_scheduler_type: str = "cosine"
    seed: int = 20260816
    max_new_tokens: int = 512


@dataclass(frozen=True)
class FineTuningConfig:
    """One reproducible experiment."""

    experiment_id: str
    base_model_id: str
    dataset_dir: Path
    dataset_version: str
    task: str = "analysis_plan"
    output_dir: Path = Path("artifacts")
    lora: LoraSettings = field(default_factory=LoraSettings)
    quantization: QuantizationSettings = field(default_factory=QuantizationSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)

    @property
    def adapter_dir(self) -> Path:
        return self.output_dir / self.experiment_id / "adapter"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dataset_dir"] = str(self.dataset_dir)
        payload["output_dir"] = str(self.output_dir)
        payload["lora"]["target_modules"] = list(self.lora.target_modules)
        return payload


def _coerce(section: type, values: Any, name: str) -> Any:
    if values is None:
        return section()
    if not isinstance(values, dict):
        raise ConfigError(f"'{name}' must be a mapping")
    known = {item.name for item in section.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    unknown = set(values) - known
    if unknown:
        raise ConfigError(f"unknown {name} keys: {sorted(unknown)}")
    if section is LoraSettings and "target_modules" in values:
        values = {**values, "target_modules": tuple(values["target_modules"])}
    return section(**values)


def config_from_mapping(payload: dict[str, Any]) -> FineTuningConfig:
    """Build a config from a parsed mapping."""
    required = {"experiment_id", "base_model_id", "dataset_dir", "dataset_version"}
    missing = required - set(payload)
    if missing:
        raise ConfigError(f"missing required configuration keys: {sorted(missing)}")

    return FineTuningConfig(
        experiment_id=str(payload["experiment_id"]),
        base_model_id=str(payload["base_model_id"]),
        dataset_dir=Path(str(payload["dataset_dir"])),
        dataset_version=str(payload["dataset_version"]),
        task=str(payload.get("task", "analysis_plan")),
        output_dir=Path(str(payload.get("output_dir", "artifacts"))),
        lora=_coerce(LoraSettings, payload.get("lora"), "lora"),
        quantization=_coerce(QuantizationSettings, payload.get("quantization"), "quantization"),
        training=_coerce(TrainingSettings, payload.get("training"), "training"),
    )


def load_config(path: Path) -> FineTuningConfig:
    """Load YAML (preferred) or JSON configuration."""
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ConfigError(
                "PyYAML is required for YAML configuration; use a .json file instead"
            ) from exc
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ConfigError("configuration must be a mapping")
    return config_from_mapping(payload)


#: Environment variable name -> where it lands in the config.
ENV_OVERRIDES: dict[str, tuple[str, str, type]] = {
    "BASE_MODEL_ID": ("", "base_model_id", str),
    "EXPERIMENT_ID": ("", "experiment_id", str),
    "DATASET_DIR": ("", "dataset_dir", Path),
    "DATASET_VERSION": ("", "dataset_version", str),
    "OUTPUT_DIR": ("", "output_dir", Path),
    "MAX_SEQ_LENGTH": ("training", "max_seq_length", int),
    "LEARNING_RATE": ("training", "learning_rate", float),
    "NUM_EPOCHS": ("training", "num_epochs", float),
    "TRAIN_BATCH_SIZE": ("training", "train_batch_size", int),
    "GRADIENT_ACCUMULATION_STEPS": ("training", "gradient_accumulation_steps", int),
    "LORA_R": ("lora", "r", int),
    "LORA_ALPHA": ("lora", "alpha", int),
    "LORA_DROPOUT": ("lora", "dropout", float),
}


def apply_env_overrides(
    config: FineTuningConfig,
    environ: dict[str, str] | None = None,
) -> FineTuningConfig:
    """Apply ``BASE_MODEL_ID``-style overrides to a loaded config."""
    source = os.environ if environ is None else environ
    top: dict[str, Any] = {}
    nested: dict[str, dict[str, Any]] = {"training": {}, "lora": {}}

    for name, (section, attribute, caster) in ENV_OVERRIDES.items():
        raw = source.get(name)
        if raw is None or raw == "":
            continue
        try:
            value = caster(raw)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{name} is not a valid {caster.__name__}") from exc
        if section:
            nested[section][attribute] = value
        else:
            top[attribute] = value

    updated = config
    if nested["training"]:
        updated = replace(updated, training=replace(updated.training, **nested["training"]))
    if nested["lora"]:
        updated = replace(updated, lora=replace(updated.lora, **nested["lora"]))
    if top:
        updated = replace(updated, **top)
    return updated
