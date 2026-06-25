"""Dataset normalization and prompt construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Example:
    uid: str
    dataset: str
    label_dist: list[float]
    prompt: str
    fields: dict[str, Any]


def build_prompt(dataset: str, record: dict[str, Any]) -> str:
    dataset = dataset.lower()
    example = record.get("example", record)
    if dataset in {"mnli", "snli"}:
        premise = example["premise"]
        hypothesis = example["hypothesis"]
        return (
            "Please determine whether the following statement is true (entailment), "
            "undetermined (neutral), or false (contradiction) given the context below "
            "and select ONE of the listed options and start your answer with a single letter.\n"
            f"Context: {premise}\n"
            f"Statement: {hypothesis}\n"
            "A. Entailment\n"
            "B. Neutral\n"
            "C. Contradiction\n"
            "Answer:"
        )
    if dataset == "anli":
        obs1 = example.get("obs1", example.get("observation_1", example.get("beginning")))
        obs2 = example.get("obs2", example.get("observation_2", example.get("ending")))
        hyp1 = example.get("hyp1", example.get("hypothesis1"))
        hyp2 = example.get("hyp2", example.get("hypothesis2"))
        if not all([obs1, obs2, hyp1, hyp2]):
            raise KeyError("ANLI records must include obs1/obs2/hyp1/hyp2 or equivalent fields.")
        return (
            "Please determine which of the two hypotheses (A or B) is more likely to "
            "explain the transition from the beginning observation to the ending "
            "observation and select ONE of the listed options and start your answer "
            "with a single letter.\n"
            f"Beginning: {obs1}\n"
            f"Ending: {obs2}\n"
            f"A. {hyp1}\n"
            f"B. {hyp2}\n"
            "Answer:"
        )
    raise ValueError(f"Unsupported dataset: {dataset}")


def normalize_record(record: dict[str, Any], dataset: str) -> Example:
    example = record.get("example", record)
    uid = str(record.get("uid", example.get("uid", "")))
    label_dist = record.get("label_dist", record.get("HJD"))
    if label_dist is None:
        raise KeyError("Record must include `label_dist` or `HJD`.")
    return Example(
        uid=uid,
        dataset=dataset.lower(),
        label_dist=[float(x) for x in label_dist],
        prompt=record.get("InputQ") or build_prompt(dataset, record),
        fields=example,
    )


def record_for_output(example: Example, reasoning: str | None = None) -> dict[str, Any]:
    row = {
        "uid": example.uid,
        "dataset": example.dataset,
        "HJD": example.label_dist,
        "InputQ": example.prompt,
        **example.fields,
    }
    if reasoning is not None:
        row["ReasoningQ"] = reasoning
    return row
