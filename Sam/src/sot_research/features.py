"""Per-text features (spaCy en_core_web_sm), cached by text sha256 under a configuration fingerprint.

Style: surface statistics, punctuation rates, function-word rates, UPOS rates, dependency-label rates, dependency
distance/depth, rates of the most frequent words (for Burrows' Delta). All rates are per token (length-normalized).
Information: named entities, numbers and number+unit quantities, sentences (kept in the LOCAL cache only, since
they are text; committed tables carry counts).

The function-word list and the Burrows most-frequent-word list are fixed from the 200 ORIGINAL T0 texts only, so
adding a translator never changes the features of texts already computed.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter

import numpy as np
import pandas as pd

from . import paths

FEATURE_VERSION = "text-features-v1"
WORD = re.compile(r"\b\w+\b")
ALPHA_WORD = re.compile(r"[A-Za-z]+(?:'[a-z]+)?")
PUNCT = [".", ",", ";", ":", "!", "?", '"', "'", "(", ")", "-", "—", "–", "“", "”", "‘",
         "’"]
UPOS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT",
        "SCONJ", "SYM", "VERB", "X"]
ENT_LABELS = {"PERSON", "ORG", "GPE", "LOC", "NORP", "FAC", "EVENT", "WORK_OF_ART", "LAW", "PRODUCT"}
UNITS = {"%", "percent", "per", "million", "billion", "trillion", "thousand", "hundred", "m", "km", "kg", "g",
         "mph", "kph", "km/h", "ft", "feet", "miles", "mile", "metres", "meters", "metre", "meter", "years", "year",
         "months", "month", "weeks", "week", "days", "day", "hours", "hour", "minutes", "minute", "seconds",
         "pounds", "dollars", "euros", "people", "tonnes", "tons", "cm", "mm", "litres", "liters", "bn", "mn", "pm",
         "am", "st", "nd", "rd", "th"}
CURRENCY = {"£": "gbp", "$": "usd", "€": "eur"}
NUMBER = re.compile(r"(?<![\w.])([£$€]?)(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(?![\d])")


def mattr(tokens: list, window: int = 50) -> float:
    """Moving-average type-token ratio (same definition as the team's src/pilot/metrics.py)."""
    if len(tokens) < 2:
        return np.nan
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    return float(np.mean([len(set(tokens[i:i + window])) / window for i in range(len(tokens) - window + 1)]))


def numbers(text: str) -> tuple:
    """Normalized numeric tokens ("11,200" -> "11200", "3.50" -> "3.5") and number+unit quantities."""
    nums, qty = [], []
    for m in NUMBER.finditer(text):
        cur, whole, frac = m.group(1), m.group(2).replace(",", ""), (m.group(3) or "").rstrip("0")
        n = whole.lstrip("0") or "0"
        n = f"{n}.{frac}" if frac else n
        nums.append(n)
        after = text[m.end():m.end() + 16].lstrip().lower()
        unit = re.match(r"(%|[a-z/]+)", after)
        u = unit.group(1) if unit and unit.group(1) in UNITS else ""
        if cur or u:
            qty.append(f"{CURRENCY.get(cur, '')}{n}{' ' + u if u else ''}".strip())
    return nums, qty


def norm_entity(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip().lower())
    s = re.sub(r"^(the|a|an) ", "", s)
    return re.sub(r"('s|’s)$", "", s).strip(" .,'\"’")


def reference_lists(t0_texts, n_function_words: int, n_mfw: int) -> dict:
    """Function words and most-frequent words (with mean/sd of their per-token rates) from the T0 corpus."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    toks = [[w.lower() for w in WORD.findall(t)] for t in t0_texts]
    counts = Counter(w for ts in toks for w in ts)
    fw = [w for w, _ in counts.most_common() if w in ENGLISH_STOP_WORDS][:n_function_words]
    mfw = [w for w, _ in counts.most_common() if w.isalpha()][:n_mfw]
    rates = np.array([[Counter(ts)[w] / max(len(ts), 1) for w in mfw] for ts in toks])
    return {"function_words": fw, "mfw": mfw, "mfw_mean": rates.mean(axis=0).tolist(),
            "mfw_sd": rates.std(axis=0, ddof=1).tolist(), "n_reference_texts": len(toks)}


def fingerprint(ref: dict) -> str:
    import spacy

    nlp_meta = {"spacy": spacy.__version__, "model": "en_core_web_sm"}
    try:
        import en_core_web_sm
        nlp_meta["model_version"] = en_core_web_sm.__version__
    except Exception:  # noqa: BLE001
        pass
    blob = json.dumps({"v": FEATURE_VERSION, "nlp": nlp_meta, "fw": ref["function_words"], "mfw": ref["mfw"]},
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def load_nlp():
    import spacy

    nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
    nlp.max_length = 2_000_000
    return nlp


def dep_labels(nlp) -> list:
    return sorted(nlp.get_pipe("parser").labels)


def _depths(doc) -> np.ndarray:
    heads = np.array([t.head.i for t in doc], dtype=np.int64)
    depth = np.full(len(heads), -1, dtype=np.int64)
    for i in range(len(heads)):
        path, j = [], i
        while depth[j] < 0 and heads[j] != j:
            path.append(j)
            j = heads[j]
        if depth[j] < 0:
            depth[j] = 0
        base = depth[j]
        for k in reversed(path):
            base += 1
            depth[k] = base
    return depth


def doc_features(doc, ref: dict, deps: list) -> dict:
    toks = [t for t in doc if not t.is_space]
    n = max(len(toks), 1)
    text = doc.text
    words = [w.lower() for w in WORD.findall(text)]
    nw = max(len(words), 1)
    alpha = [t.text.lower() for t in toks if t.is_alpha]
    sents = [s for s in doc.sents if any(not t.is_space for t in s)]
    sent_lens = [sum(1 for t in s if not t.is_space and not t.is_punct) for s in sents] or [0]
    depth = _depths(doc)
    f = {"n_tokens": len(toks), "n_words": len(words), "n_chars": len(text), "n_sentences": len(sents),
         "sent_len_mean": float(np.mean(sent_lens)), "sent_len_sd": float(np.std(sent_lens)),
         "word_len_mean": float(np.mean([len(w) for w in alpha])) if alpha else 0.0,
         "mattr": mattr(alpha), "ttr": len(set(alpha)) / max(len(alpha), 1),
         "cap_rate": sum(t.text[:1].isupper() for t in toks if t.is_alpha) / max(len(alpha), 1),
         "dep_dist_mean": float(np.mean([abs(t.i - t.head.i) for t in toks])) if toks else 0.0,
         "dep_depth_mean": float(np.mean(depth[[t.i for t in toks]])) if toks else 0.0,
         "dep_depth_max": int(depth.max()) if len(depth) else 0}
    for i, p in enumerate(PUNCT):
        f[f"punct_{i:02d}"] = text.count(p) / n
    wc = Counter(words)
    for w in ref["function_words"]:
        f[f"fw_{w}"] = wc.get(w, 0) / nw
    for w in ref["mfw"]:
        f[f"mfw_{w}"] = wc.get(w, 0) / nw
    pos = Counter(t.pos_ for t in toks)
    for p in UPOS:
        f[f"pos_{p}"] = pos.get(p, 0) / n
    dep = Counter(t.dep_ for t in toks)
    for d in deps:
        f[f"dep_{d}"] = dep.get(d, 0) / n
    ents = sorted({(norm_entity(e.text), e.label_) for e in doc.ents if e.label_ in ENT_LABELS} - {("", "")})
    ents = [e for e in ents if e[0]]
    nums, qty = numbers(text)
    f.update(n_entities=len(ents), n_numbers=len(nums), n_quantities=len(qty),
             entities_json=json.dumps(ents), numbers_json=json.dumps(nums), quantities_json=json.dumps(qty),
             sentences_json=json.dumps([s.text.strip() for s in sents]))
    return f


PUNCT_NAMES = {f"punct_{i:02d}": p for i, p in enumerate(PUNCT)}
TEXT_COLUMNS = ["entities_json", "numbers_json", "quantities_json", "sentences_json"]   # local only (contain text)


def compute(texts: pd.DataFrame, ref: dict, n_process: int = 1, log=print) -> pd.DataFrame:
    """Features for every unique text_sha256 in `texts` (columns text, text_sha256), resuming from the cache."""
    fp = fingerprint(ref)
    cache = paths.cache(f"text_features__{fp}.parquet")
    done = pd.read_parquet(cache) if cache.exists() else pd.DataFrame(columns=["text_sha256"])
    uniq = texts.drop_duplicates("text_sha256")
    todo = uniq[~uniq.text_sha256.isin(set(done.text_sha256))]
    log(f"text features [{fp}]: {len(uniq)} unique texts, {len(uniq) - len(todo)} cached, {len(todo)} to parse")
    if len(todo):
        nlp = load_nlp()
        deps = dep_labels(nlp)
        rows = []
        for k, doc in enumerate(nlp.pipe(todo.text.tolist(), batch_size=32, n_process=n_process)):
            rows.append(doc_features(doc, ref, deps))
            if (k + 1) % 250 == 0:
                log(f"  parsed {k + 1}/{len(todo)}")
        new = pd.DataFrame(rows).assign(text_sha256=todo.text_sha256.values)
        done = pd.concat([done, new], ignore_index=True) if len(done) else new
        tmp = cache.with_suffix(".tmp")
        done.to_parquet(tmp, index=False)
        tmp.replace(cache)
    return done[done.text_sha256.isin(set(uniq.text_sha256))].reset_index(drop=True)


STYLE_SCALARS = ["sent_len_mean", "sent_len_sd", "word_len_mean", "mattr", "ttr", "cap_rate", "dep_dist_mean",
                 "dep_depth_mean"]


def style_columns(df: pd.DataFrame) -> list:
    """Stylometric columns by explicit allowlist (never metadata such as is_human, round or raw lengths):
    surface scalars, punctuation, function-word, UPOS and dependency-label rates."""
    pref = ("punct_", "fw_", "pos_", "dep_")
    return [c for c in df.columns if c in STYLE_SCALARS or
            (c.startswith(pref) and c not in ("dep_dist_mean", "dep_depth_mean", "dep_depth_max"))]
