#!/usr/bin/env python
"""Generate Chain-of-Thought traces for a ChaosNLI split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


from cot_hlv.data import normalize_record, record_for_output
from cot_hlv.io import read_jsonl, write_jsonl
from cot_hlv.modeling import generate_reasoning, load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["mnli", "snli", "anli"])
    parser.add_argument("--input", required=True, help="ChaosNLI JSONL file.")
    parser.add_argument("--output", required=True, help="Output JSONL with ReasoningQ.")
    parser.add_argument("--model", required=True, help="Hugging Face model name or local model path.")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded = load_model(args.model, cache_dir=args.cache_dir, dtype=args.dtype, device_map=args.device_map)
    outputs = []
    for record in read_jsonl(args.input):
        example = normalize_record(record, args.dataset)
        reasoning = generate_reasoning(loaded, example.prompt, max_new_tokens=args.max_new_tokens)
        outputs.append(record_for_output(example, reasoning))
    write_jsonl(args.output, outputs)


if __name__ == "__main__":
    main()
