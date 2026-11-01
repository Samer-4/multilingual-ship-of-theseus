"""Output checks for one translation step. Each returns codes prefixed "check:".

Blocking (make a record invalid): missing_output, blank, unchanged, wrong_language, repetition,
  not_target_language.
  "unchanged" means ONE STEP copied its input (English left in an en->es step, or Spanish left in an
  es->en step), i.e. nothing was translated.
Warnings (recorded, not blocking): length_ratio, possible_truncation, meta_text, roundtrip_unchanged, and the
  content warnings (validation.content in experiment.yaml): sentence_drop and number_change for one step,
  roundtrip_length_loss and entity_loss for a whole round trip (English T_{r-1} -> English T_r). They point a
  reader at likely omissions or changed facts; they are heuristics and never make a record invalid.
  "roundtrip_unchanged" means T_r is identical to the English that entered the round trip. That can be a
  legitimate, perfectly faithful round trip, so it is never a failure.
Engine codes ("engine:...") are added by the runner: generation_cap (some output hit max_new_tokens without
EOS), empty_segment, segmentation_error. All are blocking. Nothing here ever edits the generated text.
The language heuristic and thresholds come from experiments/qwen_translation/degree_zero_chain.ipynb.
"wrong_language" only detects the SOURCE language left in the output. "not_target_language" (added after the
first Kaggle MADLAD runs) requires the output to look like the target language at all: few non-Latin letters,
and some target function words in any output of >= min_words words. A mis-wired model produced EOS-terminated
garbage ("εαααεόσwwnw ...") that passed every other check. Thresholds: experiment.yaml validation.target_language.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

EN_RATE = re.compile(r"\b(the|and|of|was|is|that|with)\b", re.I)
ES_RATE = re.compile(r"\b(el|los|las|del|que|y|una|por|con|es)\b", re.I)
TERMINAL = re.compile(r"""[.!?…"'”’)\]]\s*$""")
META_TEXT = re.compile(r"^\s*(here is|here's|sure[,!]|translation:|aquí (está|tienes)|traducción:)", re.I)
BLOCKING = {"missing_output", "blank", "unchanged", "wrong_language", "repetition", "not_target_language",
            "generation_cap", "empty_segment", "segmentation_error"}


# Validation keys introduced after Stage 2. Legacy imports (Stage 2 MADLAD, legacy Qwen) are judged by the rules
# that produced the staged table, so rebuilding Stage 2 stays byte-identical; new runs use all keys.
ADDED_AFTER_STAGE2 = ("target_language", "content")


def stage2_rules(validation_cfg: dict) -> dict:
    return {k: v for k, v in validation_cfg.items() if k not in ADDED_AFTER_STAGE2}


def _same(a: str, b: str) -> bool:
    return " ".join(str(a).split()) == " ".join(str(b).split())


def roundtrip_warnings(english_before: str, english_after) -> list:
    """For a T stage: identical English after a full round trip is recorded, not rejected."""
    if isinstance(english_after, str) and english_after.strip() and _same(english_before, english_after):
        return ["check:roundtrip_unchanged"]
    return []


SENTENCE_END = re.compile(r"[.!?…]+[\"'”’)\]]*(?=\s|$)")
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
NAME = re.compile(r"\b[A-Z][a-zA-Z\-]{2,}")          # >= 3 letters: skips PS (XSum mojibake), TV, UK, Mr


def sentence_count(text: str) -> int:
    t = str(text).strip()
    return len(SENTENCE_END.findall(t)) + (0 if not t or SENTENCE_END.search(t[-3:]) else 1)


def numbers(text: str) -> set:
    """Digit groups with separators removed, so 400,000 / 400.000 and 1.4 / 1,4 compare equal."""
    return {re.sub(r"[.,]", "", n) for n in NUMBER.findall(str(text))}


def names(text: str) -> set:
    """Capitalised words that do not start a sentence (a cheap named-entity proxy), lower-cased."""
    out, start = set(), True
    for tok in re.findall(r"\S+", str(text)):
        w = NAME.match(tok.lstrip("\"'“‘(["))
        if w and not start:
            out.add(w.group(0).lower())
        start = bool(SENTENCE_END.search(tok))
    return out


def content_warnings(input_text: str, output_text: str, cfg: dict) -> list:
    """Step-level: fewer sentences than the input; numbers lost or added."""
    c = (cfg or {}).get("content")
    if not c or not isinstance(output_text, str):
        return []
    warn = []
    n_in, n_out = sentence_count(input_text), sentence_count(output_text)
    if n_in >= c["sentence_drop_min_input"] and n_out < n_in:
        warn.append(f"check:sentence_drop={n_in}>{n_out}")
    a, b = numbers(input_text), numbers(output_text)
    if a != b:
        k = c["max_listed"]
        warn.append("check:number_change=lost:" + "|".join(sorted(a - b)[:k]) + "/added:" + "|".join(sorted(b - a)[:k]))
    return warn


def roundtrip_content_warnings(english_before: str, english_after, cfg: dict) -> list:
    """Round-trip level, same language: substantial length loss; names present before and gone after."""
    c = (cfg or {}).get("content")
    if not c or not isinstance(english_after, str) or not english_after.strip():
        return []
    warn = []
    ratio = len(english_after.split()) / max(1, len(str(english_before).split()))
    if ratio < c["roundtrip_length_loss_below"]:
        warn.append(f"check:roundtrip_length_loss={ratio:.2f}")
    lost = sorted(names(english_before) - names(english_after) - {w.lower() for w in re.findall(r"[A-Za-z\-]+", english_after)})
    if lost:
        warn.append("check:entity_loss=" + "|".join(lost[:c["max_listed"]]) + ("|..." if len(lost) > c["max_listed"] else ""))
    return warn


def _rate(pattern, text: str) -> float:
    return len(pattern.findall(text)) / max(1, len(text.split()))


def non_latin_share(text: str) -> float:
    letters = [c for c in str(text) if c.isalpha()]
    return sum("LATIN" not in unicodedata.name(c, "") for c in letters) / max(1, len(letters))


def target_rate(text: str, lang: str) -> float:
    return _rate(ES_RATE if lang == "es" else EN_RATE, str(text))


def repetition(text: str, count_min: int, share_min: float) -> bool:
    """A decoding loop: one word 3-gram repeated >= count_min times and covering >= share_min of all 3-grams."""
    w = re.findall(r"\w+", text.lower())
    grams = list(zip(w, w[1:], w[2:]))
    if not grams:
        return False
    top = Counter(grams).most_common(1)[0][1]
    return top >= count_min and top / len(grams) >= share_min


def check_step(input_text: str, output_text, output_language: str, cfg: dict) -> dict:
    """Return {"blocking": [...], "warnings": [...]} for one step (input -> output)."""
    block, warn = [], []
    if output_text is None or (isinstance(output_text, float) and output_text != output_text):
        return {"blocking": ["check:missing_output"], "warnings": []}
    out = str(output_text)
    if not out.strip():
        return {"blocking": ["check:blank"], "warnings": []}
    if _same(out, input_text):
        block.append("check:unchanged")
    lang = cfg["wrong_language"]
    if output_language == "es" and _rate(EN_RATE, out) > lang["english_rate_in_spanish_max"]:
        block.append("check:wrong_language")
    if output_language == "en" and _rate(ES_RATE, out) > lang["spanish_rate_in_english_max"]:
        block.append("check:wrong_language")
    tl = cfg.get("target_language")
    if tl and output_language in {"es", "en"}:
        if non_latin_share(out) > tl["non_latin_letter_share_max"] or (
                len(out.split()) >= tl["min_words"] and target_rate(out, output_language) < tl["function_word_rate_min"]):
            block.append("check:not_target_language")
    rep = cfg["repetition"]
    if repetition(out, rep["trigram_count_min"], rep["trigram_share_min"]):
        block.append("check:repetition")
    lo, hi = cfg["step_word_ratio_warn"]
    ratio = len(out.split()) / max(1, len(str(input_text).split()))
    if not lo <= ratio <= hi:
        warn.append(f"check:length_ratio={ratio:.2f}")
    if TERMINAL.search(str(input_text)) and not TERMINAL.search(out):
        warn.append("check:possible_truncation")
    if META_TEXT.search(out) and not META_TEXT.search(str(input_text)):
        warn.append("check:meta_text")
    warn += content_warnings(input_text, out, cfg)
    return {"blocking": block, "warnings": warn}
