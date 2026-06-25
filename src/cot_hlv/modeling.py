"""Model loading and generation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LoadedModel:
    name_or_path: str
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer


def load_model(
    model_name_or_path: str,
    cache_dir: str | None = None,
    dtype: str = "bfloat16",
    device_map: str = "auto",
    trust_remote_code: bool = True,
) -> LoadedModel:
    torch_dtype = getattr(torch, dtype) if dtype != "auto" else "auto"
    path = str(Path(model_name_or_path).expanduser())
    model = AutoModelForCausalLM.from_pretrained(
        path,
        cache_dir=cache_dir,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )
    return LoadedModel(path, model, tokenizer)


def chat_prompt(tokenizer: AutoTokenizer, user_prompt: str) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_reasoning(decoded_text: str) -> str:
    """Extract a reasoning span from common open reasoning model templates."""
    markers = [
        ("<|channel|>analysis<|message|>", "<|end|><|start|>assistant<|channel|>final<|message|>"),
        ("<think>", "</think>"),
        ("", "\n</think>\n"),
    ]
    for start, end in markers:
        segment = decoded_text
        if start and start in segment:
            segment = segment.split(start, 1)[1]
        if end in segment:
            return segment.split(end, 1)[0].strip()
    return decoded_text.strip()


@torch.no_grad()
def generate_reasoning(
    loaded: LoadedModel,
    prompt: str,
    max_new_tokens: int = 32768,
) -> str:
    text = chat_prompt(loaded.tokenizer, prompt)
    inputs = loaded.tokenizer([text], return_tensors="pt").to(loaded.model.device)
    output_ids = loaded.model.generate(**inputs, max_new_tokens=max_new_tokens)
    decoded = loaded.tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0]
    return extract_reasoning(decoded)


def option_token_ids(tokenizer: AutoTokenizer, options: Sequence[str]) -> list[int]:
    ids = []
    for option in options:
        token_ids = tokenizer.encode(option, add_special_tokens=False)
        if len(token_ids) != 1:
            spaced = tokenizer.encode(" " + option, add_special_tokens=False)
            if len(spaced) == 1:
                token_ids = spaced
        if len(token_ids) != 1:
            raise ValueError(
                f"Option {option!r} does not map to one token. Pass explicit token IDs instead."
            )
        ids.append(token_ids[0])
    return ids


def build_injected_prompt(
    tokenizer: AutoTokenizer,
    user_prompt: str,
    cot_prefix: str,
    template: str = "think",
) -> str:
    base = chat_prompt(tokenizer, user_prompt)
    if template == "gpt-oss":
        return (
            base
            + "<|channel|>analysis<|message|>"
            + cot_prefix
            + "\n<|end|><|start|>assistant<|channel|>final<|message|>\n\n"
            + "Based on the reasoning so far, the Answer is:"
        )
    if template == "plain":
        return base + cot_prefix + "\n\nBased on the reasoning so far, the Answer is:"
    return base + cot_prefix + "\n</think>\n\nBased on the reasoning so far, the Answer is:"


@torch.no_grad()
def first_step_option_logits(
    loaded: LoadedModel,
    input_text: str,
    option_ids: Sequence[int],
    max_new_tokens: int = 20,
) -> tuple[str, list[float]]:
    inputs = loaded.tokenizer([input_text], return_tensors="pt").to(loaded.model.device)
    generated = loaded.model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_logits=True,
        do_sample=False,
    )
    answer_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated.sequences)
    ]
    answer = loaded.tokenizer.batch_decode(answer_ids, skip_special_tokens=True)[0]
    logits = generated.logits[0][0]
    return answer, [float(logits[token_id].detach().cpu()) for token_id in option_ids]
