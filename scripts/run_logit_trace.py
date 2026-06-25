#!/usr/bin/env python
"""Inject CoT prefixes and collect first-token option logits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


from cot_hlv.io import read_jsonl, write_jsonl
from cot_hlv.modeling import load_model, option_token_ids
from cot_hlv.trace import run_trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSONL containing InputQ and chunks.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True, help="Hugging Face model name or local model path.")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--template", choices=["think", "gpt-oss", "plain"], default="think")
    parser.add_argument("--options", nargs="+", default=["A", "B", "C"])
    parser.add_argument("--option-token-ids", nargs="+", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded = load_model(args.model, cache_dir=args.cache_dir, dtype=args.dtype, device_map=args.device_map)
    ids = args.option_token_ids or option_token_ids(loaded.tokenizer, args.options)
    rows = read_jsonl(args.input)
    traced = run_trace(
        loaded,
        rows,
        option_ids=ids,
        template=args.template,
        max_new_tokens=args.max_new_tokens,
    )
    write_jsonl(args.output, traced)


if __name__ == "__main__":
    main()
