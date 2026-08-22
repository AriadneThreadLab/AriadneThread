"""Supervised fine-tuning with TRL's ``SFTTrainer``.

Only QLoRA + SFT is implemented. DPO is a plausible next step once the review
queue holds enough approved/rejected pairs for the *same* request, but pairs of
that kind do not exist yet, so it is documented rather than written.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ariadne_finetuning.config import FineTuningConfig
from ariadne_finetuning.dataset import ChatSample, DatasetBundle
from ariadne_finetuning.modeling import (
    SYSTEM_INSTRUCTION,
    build_lora_config,
    load_base_model,
    load_tokenizer,
    require_dependencies,
)


def to_conversation(sample: ChatSample) -> dict[str, Any]:
    """Chat-template rows for TRL."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": sample.prompt},
            {"role": "assistant", "content": sample.completion},
        ]
    }


def build_hf_datasets(bundle: DatasetBundle) -> tuple[Any, Any]:
    """Convert the exported splits into Hugging Face datasets."""
    require_dependencies()
    from datasets import Dataset

    train = Dataset.from_list([to_conversation(sample) for sample in bundle.train])
    validation = (
        Dataset.from_list([to_conversation(sample) for sample in bundle.validation])
        if bundle.validation
        else None
    )
    return train, validation


def train_adapter(
    config: FineTuningConfig,
    bundle: DatasetBundle,
    *,
    output_dir: Path | None = None,
) -> Path:
    """Run QLoRA SFT and save the adapter. Returns the adapter directory.

    Nothing is deployed: the caller gets a path, and promotion stays a manual
    decision recorded in the experiment registry.
    """
    require_dependencies()
    from trl import SFTConfig, SFTTrainer

    adapter_dir = output_dir or config.adapter_dir
    adapter_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(config)
    model = load_base_model(config, for_training=True)
    train_dataset, eval_dataset = build_hf_datasets(bundle)

    settings = config.training
    sft_config = SFTConfig(
        output_dir=str(adapter_dir.parent / "checkpoints"),
        per_device_train_batch_size=settings.train_batch_size,
        per_device_eval_batch_size=settings.eval_batch_size,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        gradient_checkpointing=settings.gradient_checkpointing,
        learning_rate=settings.learning_rate,
        num_train_epochs=settings.num_epochs,
        warmup_ratio=settings.warmup_ratio,
        weight_decay=settings.weight_decay,
        lr_scheduler_type=settings.lr_scheduler_type,
        logging_steps=settings.logging_steps,
        max_length=settings.max_seq_length,
        optim=settings.optim,
        seed=settings.seed,
        report_to=[],
        save_strategy="epoch",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=build_lora_config(config),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    write_adapter_metadata(config, bundle, adapter_dir)
    return adapter_dir


def write_adapter_metadata(
    config: FineTuningConfig,
    bundle: DatasetBundle,
    adapter_dir: Path,
) -> Path:
    """Save provenance beside the adapter weights."""
    path = adapter_dir / "ariadne_adapter_metadata.json"
    path.write_text(
        json.dumps(
            {
                "experiment_id": config.experiment_id,
                "base_model_id": config.base_model_id,
                "task": config.task,
                "dataset_version": bundle.dataset_version,
                "test_digest": bundle.test_digest,
                "train_examples": len(bundle.train),
                "training_config": config.to_dict(),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path
