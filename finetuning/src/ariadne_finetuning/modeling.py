"""Base-model loading and generation (QLoRA).

Every machine-learning import happens *inside* a function. Importing this
module therefore costs nothing and requires no GPU, which is what keeps the
rest of the package testable offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ariadne_finetuning.config import FineTuningConfig

#: Packages required to actually train or run a model.
TRAINING_REQUIREMENTS: tuple[str, ...] = (
    "torch",
    "transformers",
    "peft",
    "trl",
    "bitsandbytes",
    "accelerate",
    "datasets",
)

SYSTEM_INSTRUCTION = (
    "You are Ariadne Thread's planner. Given a GIS request, return exactly one "
    "JSON object matching the requested plan schema. No prose, no Markdown, no "
    "explanation."
)


class MissingDependencyError(RuntimeError):
    """The fine-tuning stack is not installed in this environment."""


def missing_requirements() -> list[str]:
    """Which training packages cannot be imported here."""
    import importlib.util

    return [name for name in TRAINING_REQUIREMENTS if importlib.util.find_spec(name) is None]


def require_dependencies() -> None:
    """Fail with an actionable message instead of an ImportError traceback."""
    missing = missing_requirements()
    if missing:
        raise MissingDependencyError(
            "missing fine-tuning dependencies: "
            + ", ".join(missing)
            + ". Install them with: pip install -e 'finetuning[train]'"
        )


def describe_environment() -> dict[str, Any]:
    """GPU/dependency probe. Colab does not always hand out a GPU."""
    info: dict[str, Any] = {
        "missing_requirements": missing_requirements(),
        "cuda_available": False,
        "device_name": None,
        "total_memory_gb": None,
    }
    try:
        import torch
    except ImportError:
        return info
    info["cuda_available"] = bool(torch.cuda.is_available())
    if info["cuda_available"]:
        properties = torch.cuda.get_device_properties(0)
        info["device_name"] = properties.name
        info["total_memory_gb"] = round(properties.total_memory / (1024**3), 2)
    return info


def build_quantization_config(config: FineTuningConfig) -> Any:
    """4-bit NF4 with double quantization (the QLoRA recipe)."""
    require_dependencies()
    import torch
    from transformers import BitsAndBytesConfig

    dtype = getattr(torch, config.quantization.compute_dtype, torch.float16)
    return BitsAndBytesConfig(
        load_in_4bit=config.quantization.load_in_4bit,
        bnb_4bit_quant_type=config.quantization.quant_type,
        bnb_4bit_use_double_quant=config.quantization.double_quant,
        bnb_4bit_compute_dtype=dtype,
    )


def build_lora_config(config: FineTuningConfig) -> Any:
    """PEFT configuration for the adapter."""
    require_dependencies()
    from peft import LoraConfig

    return LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=list(config.lora.target_modules),
        bias=config.lora.bias,
        task_type=config.lora.task_type,
    )


def load_tokenizer(config: FineTuningConfig) -> Any:
    require_dependencies()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.base_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_base_model(config: FineTuningConfig, *, for_training: bool = True) -> Any:
    """Load the base model in 4 bits, optionally prepared for k-bit training."""
    require_dependencies()
    from peft import prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_id,
        quantization_config=build_quantization_config(config),
        device_map="auto",
    )
    if for_training:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=config.training.gradient_checkpointing,
        )
        model.config.use_cache = False
    return model


@dataclass
class TransformersRunner:
    """A :class:`~ariadne_finetuning.evaluation.ModelRunner` over a real model."""

    model: Any
    tokenizer: Any
    max_new_tokens: int = 512
    model_label: str = "base"

    @property
    def label(self) -> str:
        return self.model_label

    def generate(self, prompt: str) -> str:
        import torch

        messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                # Greedy decoding: evaluation must be repeatable.
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[-1] :]
        return str(self.tokenizer.decode(generated, skip_special_tokens=True))


def load_runner(
    config: FineTuningConfig,
    *,
    adapter_path: Path | None = None,
    label: str | None = None,
) -> TransformersRunner:
    """Build a runner for the base model, or for the base model plus adapter."""
    require_dependencies()
    model = load_base_model(config, for_training=False)
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.eval()
    return TransformersRunner(
        model=model,
        tokenizer=load_tokenizer(config),
        max_new_tokens=config.training.max_new_tokens,
        model_label=label or (config.base_model_id if adapter_path is None else "finetuned"),
    )
