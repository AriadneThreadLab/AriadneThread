#!/usr/bin/env python
"""Record a human promotion decision for a trained adapter.

    python scripts/promote.py --registry artifacts/registry.json \
        --experiment-id analysis-plan-qlora-001 --status approved \
        --notes "beats baseline on target preservation; reviewed by <name>"

Marking an experiment ``approved`` records a decision. It does not change any
running service: wiring an adapter into Ariadne is a deliberate, separate act.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ariadne_finetuning.registry import ExperimentRegistry, ExperimentStatus


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("artifacts/registry.json"))
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument(
        "--status",
        required=True,
        choices=[status.value for status in ExperimentStatus],
    )
    parser.add_argument("--notes", default="")
    args = parser.parse_args(argv)

    registry = ExperimentRegistry(args.registry)
    try:
        record = registry.set_status(
            args.experiment_id,
            ExperimentStatus(args.status),
            notes=args.notes,
        )
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(record.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
