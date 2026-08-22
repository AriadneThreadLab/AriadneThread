#!/usr/bin/env python
"""Run the full pipeline: validate, baseline, QLoRA SFT, evaluate, record.

    python scripts/train_qlora.py --config configs/qlora_analysis_plan.yaml

Nothing is deployed. The output is an adapter directory, a before/after report
and an ``experimental`` entry in the registry; promoting it into Ariadne is a
separate, manual decision.

Requires the training extra: pip install -e '.[train]'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ariadne_finetuning.config import (
    FineTuningConfig,
    apply_env_overrides,
    load_config,
)
from ariadne_finetuning.dataset import DatasetBundle
from ariadne_finetuning.evaluation import ModelRunner
from ariadne_finetuning.modeling import describe_environment, load_runner
from ariadne_finetuning.pipeline import run_pipeline
from ariadne_finetuning.registry import ExperimentRegistry
from ariadne_finetuning.training import train_adapter

REPO_ROOT = Path(__file__).resolve().parents[2]


def runner_factory(config: FineTuningConfig, adapter: Path | None) -> ModelRunner:
    return load_runner(
        config,
        adapter_path=adapter,
        label="finetuned" if adapter else config.base_model_id,
    )


def trainer(config: FineTuningConfig, bundle: DatasetBundle) -> Path:
    return train_adapter(config, bundle)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Measure the base model and stop before training.",
    )
    args = parser.parse_args(argv)

    config = apply_env_overrides(load_config(args.config))
    environment = describe_environment()
    print(json.dumps({"environment": environment}, indent=2), file=sys.stderr)
    if not args.baseline_only and not environment["cuda_available"]:
        print(
            "warning: no CUDA device detected. QLoRA training on CPU is not practical.",
            file=sys.stderr,
        )

    result = run_pipeline(
        config,
        runner_factory=runner_factory,
        trainer=trainer,
        registry=ExperimentRegistry(config.output_dir / "registry.json"),
        baseline_only=args.baseline_only,
        repo_root=REPO_ROOT,
    )

    summary = json.dumps(result.summary(), indent=2)
    print(summary)
    report_path = config.output_dir / config.experiment_id / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(summary + "\n", encoding="utf-8")
    print(f"\nreport: {report_path}", file=sys.stderr)
    print("status: experimental (promotion is a separate manual step)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
