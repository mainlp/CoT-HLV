#!/usr/bin/env python
"""Evaluate traced logits against human label distributions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


from cot_hlv.io import read_jsonl
from cot_hlv.metrics import aggregate_metrics, evaluate_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Trace JSONL containing HJD and logits.")
    parser.add_argument("--output", default=None, help="Optional CSV output path.")
    parser.add_argument("--probability-mode", choices=["linear", "softmax"], default="linear")
    parser.add_argument("--aggregate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = evaluate_rows(read_jsonl(args.input), probability_mode=args.probability_mode)
    if args.aggregate:
        frame = aggregate_metrics(frame)
    if args.output:
        frame.to_csv(args.output, index=False)
    else:
        print(frame.to_csv(index=False))


if __name__ == "__main__":
    main()
