#!/usr/bin/env python
"""Evaluate the base model, or a trained adapter, on a frozen test set.

    # baseline, before any training
    python scripts/evaluate.py --config configs/qlora_analysis_plan.yaml

    # the same examples, now with an adapter
    python scripts/evaluate.py --config configs/qlora_analysis_plan.yaml \
        --adapter artifacts/analysis-plan-qlora-001/adapter

    # the hand-curated gold set instead of the exported test split
    python scripts/evaluate.py --config configs/qlora_analysis_plan.yaml \
        --gold evaluation/gold/analysis_plan_gold_v1

Requires the training extra: pip install -e '.[train]'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ariadne_finetuning.config import apply_env_overrides, load_config
from ariadne_finetuning.dataset import load_bundle, load_jsonl
from ariadne_finetuning.evaluation import evaluate_samples
from ariadne_finetuning.modeling import describe_environment, load_runner


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument(
        "--gold",
        type=Path,
        default=None,
        help="Gold-set stem, without the .test.jsonl / .schema.json suffix.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    config = apply_env_overrides(load_config(args.config))
    print(json.dumps({"environment": describe_environment()}, indent=2), file=sys.stderr)

    if args.gold is not None:
        samples = load_jsonl(args.gold.with_suffix(".test.jsonl"))
        schema = json.loads(args.gold.with_suffix(".schema.json").read_text(encoding="utf-8"))
        version = args.gold.name
        digest = f"gold:{args.gold.name}"
    else:
        bundle = load_bundle(
            config.dataset_dir, task=config.task, dataset_version=config.dataset_version
        )
        samples, schema = bundle.test, bundle.target_schema
        version, digest = bundle.dataset_version, bundle.test_digest

    runner = load_runner(
        config,
        adapter_path=args.adapter,
        label="finetuned" if args.adapter else config.base_model_id,
    )
    report = evaluate_samples(
        runner,
        samples,
        schema=schema,
        dataset_version=version,
        test_digest=digest,
    )

    payload = json.dumps(report.as_dict(), indent=2)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
