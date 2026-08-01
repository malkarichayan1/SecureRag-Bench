"""Summarize native InjecAgent artifacts and compare execution ASR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from secure_rag_bench.evaluation.native_analysis import analyze_native_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Native InjecAgent JSON artifacts")
    parser.add_argument("--output", required=True, help="Path for the summary JSON artifact")
    parser.add_argument(
        "--paired-inference",
        action="store_true",
        help="Run exact case-paired McNemar tests with Holm family correction",
    )
    args = parser.parse_args()

    report = analyze_native_artifacts(
        [Path(path) for path in args.inputs],
        paired_inference=args.paired_inference,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
