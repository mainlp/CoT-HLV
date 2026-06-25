#!/usr/bin/env python
"""Create cumulative CoT prefixes for step-wise tracing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


from cot_hlv.chunking import cumulative_chunks
from cot_hlv.io import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL containing ReasoningQ.")
    parser.add_argument("--output", required=True, help="Output JSONL containing chunks.")
    parser.add_argument("--num-chunks", type=int, default=10)
    parser.add_argument("--method", choices=["token", "sentence", "char"], default="token")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for row in read_jsonl(args.input):
        out = dict(row)
        out["chunks"] = cumulative_chunks(
            row.get("ReasoningQ", ""),
            num_chunks=args.num_chunks,
            method=args.method,
        )
        rows.append(out)
    write_jsonl(args.output, rows)


if __name__ == "__main__":
    main()
