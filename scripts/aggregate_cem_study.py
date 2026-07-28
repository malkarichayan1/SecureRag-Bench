"""Merge independently generated CEM seed artifacts into one study artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from secure_rag_bench.evaluation.offline_study import aggregate_cem_studies


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate CEM seed study artifacts")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    studies = [json.loads(path.read_text(encoding="utf-8"))["cem_study"] for path in args.inputs]
    result = aggregate_cem_studies(studies)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
