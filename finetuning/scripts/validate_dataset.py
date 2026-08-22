#!/usr/bin/env python
"""Validate an exported dataset before spending any GPU time.

python scripts/validate_dataset.py --config configs/qlora_analysis_plan.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ariadne_finetuning.config import apply_env_overrides, load_config
from ariadne_finetuning.dataset import load_bundle, validate_bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)

    config = apply_env_overrides(load_config(args.config))
    bundle = load_bundle(
        config.dataset_dir,
        task=config.task,
        dataset_version=config.dataset_version,
    )
    problems = validate_bundle(bundle)

    print(
        json.dumps(
            {
                "dataset_version": bundle.dataset_version,
                "task": bundle.task,
                "counts": {
                    "train": len(bundle.train),
                    "validation": len(bundle.validation),
                    "test": len(bundle.test),
                },
                "test_digest": bundle.test_digest,
                "target_schema": bundle.manifest.get("target_schema_name"),
                "problems": problems,
            },
            indent=2,
        )
    )
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
