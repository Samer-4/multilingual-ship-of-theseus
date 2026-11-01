"""Deterministic document segmentation for the MT models, and reconstruction of their outputs.

  madlad_pack  exact replica of split_chunks() in the MADLAD worker (madlad_explorer.ipynb, cell 5):
               regex sentence split, word-level fallback for over-long sentences, then greedy packing of
               consecutive pieces into chunks of <= budget tokens (prefix token included in the count).
  sentence     the same sentence split and word fallback, but one sentence per segment (no packing), for
               NLLB, which was trained on sentence-level pairs.
  none         the whole document is one segment (Qwen, Llama).
Reconstruction joins segment outputs with a single space, as the worker did.

Confound: MT pathways are segmented and LLM pathways are not, so sentence boundaries are preserved by
construction for MADLAD/NLLB and not for the LLMs. Sentence-length and punctuation style metrics, and
translator-fingerprint classifiers, partly reflect this design choice.
"""
from __future__ import annotations

import re
from typing import Callable, List

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
JOINER = " "


class SegmentationError(ValueError):
    """Deterministic: the same input will always fail (e.g. one word longer than the token budget)."""


def _pieces(text: str, count: Callable[[str], int], budget: int) -> List[str]:
    pieces = []
    for sentence in SENTENCE_SPLIT.split(str(text).strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        if count(sentence) <= budget:
            pieces.append(sentence)
            continue
        part = ""
        for word in sentence.split():
            trial = (part + " " + word).strip()
            if count(trial) <= budget:
                part = trial
            else:
                if part:
                    pieces.append(part)
                if count(word) > budget:
                    raise SegmentationError("one_word_exceeds_budget")
                part = word
        if part:
            pieces.append(part)
    return pieces


def madlad_pack(text: str, count: Callable[[str], int], budget: int) -> List[str]:
    result, cur = [], ""
    for part in _pieces(text, count, budget):
        candidate = (cur + " " + part).strip()
        if count(candidate) <= budget:
            cur = candidate
        else:
            if cur:
                result.append(cur)
            cur = part
    if cur:
        result.append(cur)
    if not result:
        raise SegmentationError("no_input_chunks")
    return result


def sentences(text: str, count: Callable[[str], int], budget: int) -> List[str]:
    result = _pieces(text, count, budget)
    if not result:
        raise SegmentationError("no_input_chunks")
    return result


def segment(text: str, method: str, count: Callable[[str], int], budget) -> List[str]:
    if method == "madlad_pack":
        return madlad_pack(text, count, budget)
    if method == "sentence":
        return sentences(text, count, budget)
    if method == "none":
        if not str(text).strip():
            raise SegmentationError("no_input_chunks")
        return [str(text)]
    raise ValueError(f"unknown segmentation method {method!r}")


def reconstruct(outputs: List[str]) -> str:
    return JOINER.join(outputs)


def describe(segments: List[str], method: str, count: Callable[[str], int], budget) -> dict:
    """Metadata stored with each generation: how the input was cut and how the output was joined."""
    return {"method": method, "budget_tokens": budget, "n_segments": len(segments), "joiner": JOINER,
            "segment_input_tokens": [count(s) for s in segments],
            "segment_input_words": [len(s.split()) for s in segments]}
