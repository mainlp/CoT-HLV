"""Cross-CoT logit tracing."""

from __future__ import annotations

from typing import Sequence

from tqdm import tqdm

from .modeling import LoadedModel, build_injected_prompt, first_step_option_logits


def run_trace(
    loaded: LoadedModel,
    rows: list[dict],
    option_ids: Sequence[int],
    template: str = "think",
    max_new_tokens: int = 20,
) -> list[dict]:
    outputs = []
    for row in tqdm(rows, desc="Tracing logits"):
        traced = dict(row)
        traced["answers"] = []
        traced["logits"] = []
        for chunk in row["chunks"]:
            input_text = build_injected_prompt(
                loaded.tokenizer,
                row["InputQ"],
                chunk,
                template=template,
            )
            answer, logits = first_step_option_logits(
                loaded,
                input_text,
                option_ids,
                max_new_tokens=max_new_tokens,
            )
            traced["answers"].append(answer)
            traced["logits"].append(logits)
        traced.pop("chunks", None)
        outputs.append(traced)
    return outputs
