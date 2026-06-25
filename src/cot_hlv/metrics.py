"""Evaluation metrics and ANOVA summaries."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr


def normalize_logits(logits: list[float], mode: str = "linear") -> np.ndarray:
    values = np.asarray(logits, dtype=float)
    if mode == "softmax":
        values = values - np.max(values)
        probs = np.exp(values)
        return probs / probs.sum()
    if mode != "linear":
        raise ValueError("mode must be `linear` or `softmax`.")
    values = values - values.min()
    if values.sum() <= 0:
        return np.ones_like(values) / len(values)
    return values / values.sum()


def accuracy(pred: np.ndarray, gold: np.ndarray) -> float:
    return float(np.argmax(pred) == np.argmax(gold))


def jsd(pred: np.ndarray, gold: np.ndarray) -> float:
    return float(jensenshannon(pred, gold, base=2.0))


def spearman(pred: np.ndarray, gold: np.ndarray) -> float:
    if len(pred) < 3:
        return accuracy(pred, gold)
    value = spearmanr(pred, gold).statistic
    return 0.0 if math.isnan(value) else float(value)


def evaluate_rows(rows: list[dict[str, Any]], probability_mode: str = "linear") -> pd.DataFrame:
    records = []
    for row in rows:
        gold = np.asarray(row.get("HJD", row.get("label_dist")), dtype=float)
        logits_steps = row.get("logits")
        if logits_steps is None:
            logits_steps = [row["logits_start"], row["logits_last"]]
        for step, logits in enumerate(logits_steps):
            pred = normalize_logits(logits, probability_mode)
            records.append(
                {
                    "uid": row.get("uid"),
                    "dataset": row.get("dataset"),
                    "step": step,
                    "accuracy": accuracy(pred, gold),
                    "jsd": jsd(pred, gold),
                    "spearman": spearman(pred, gold),
                }
            )
    return pd.DataFrame.from_records(records)


def aggregate_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["accuracy", "jsd", "spearman"]
    group_cols = [col for col in ["dataset", "model", "cot_source", "step"] if col in frame.columns]
    if not group_cols:
        group_cols = ["step"]
    return frame.groupby(group_cols, dropna=False)[metric_cols].mean().reset_index()


def anova_contributions(
    frame: pd.DataFrame,
    score_col: str,
    model_col: str = "model",
    cot_col: str = "cot_source",
) -> pd.DataFrame:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    data = frame[[score_col, model_col, cot_col]].dropna().rename(
        columns={score_col: "score", model_col: "model", cot_col: "cot"}
    )
    fit = ols("score ~ C(model) + C(cot)", data=data).fit()
    table = sm.stats.anova_lm(fit, typ=2).reset_index().rename(columns={"index": "factor"})
    total = table["sum_sq"].sum()
    table["contribution_pct"] = table["sum_sq"] / total * 100.0
    return table
