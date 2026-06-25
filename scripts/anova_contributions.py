#!/usr/bin/env python
"""Compute additive two-way ANOVA contribution percentages."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


import pandas as pd

from cot_hlv.metrics import anova_contributions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV with score, model, and cot_source columns.")
    parser.add_argument("--score", required=True, help="Metric column, e.g. accuracy/jsd/spearman.")
    parser.add_argument("--model-col", default="model")
    parser.add_argument("--cot-col", default="cot_source")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = anova_contributions(
        pd.read_csv(args.input),
        score_col=args.score,
        model_col=args.model_col,
        cot_col=args.cot_col,
    )
    if args.output:
        table.to_csv(args.output, index=False)
    else:
        print(table.to_csv(index=False))


if __name__ == "__main__":
    main()
