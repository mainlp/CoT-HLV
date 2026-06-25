"""Utilities for cumulative CoT truncation."""

from __future__ import annotations

import re


def sentence_split(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [part.strip() for part in parts if part.strip()]


def cumulative_chunks(text: str, num_chunks: int = 10, method: str = "token") -> list[str]:
    """Return progressively longer prefixes from 0% to 100% of a reasoning trace."""
    if num_chunks < 1:
        raise ValueError("num_chunks must be positive.")
    text = text.strip()
    if not text:
        return [""]

    if method == "sentence":
        units = sentence_split(text)
    elif method == "char":
        units = list(text)
    elif method == "token":
        units = text.split()
    else:
        raise ValueError("method must be one of: token, sentence, char.")

    if not units:
        return [text]

    chunks = []
    for i in range(num_chunks + 1):
        end = round(len(units) * i / num_chunks)
        if method == "char":
            chunk = "".join(units[:end])
        else:
            chunk = " ".join(units[:end])
        chunks.append(chunk.strip())
    return chunks
