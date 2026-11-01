"""Pairwise metrics (ref -> hyp) and the pair list. CPU metrics are cached by (ref_sha256, hyp_sha256).

Pairs (English only, both texts valid):
  cumulative  T0 vs Tk (k = 1..3), per translator
  step        T(k-1) vs Tk (k = 1..3), per translator; step k=1 is the same pair as cumulative k=1
  baseline    translator-free reference points: "unrelated" (a T0 vs the T0 of a different article, fixed seeded
              assignment) and "cross_source" (an LLM-source T0 vs the Human T0 of the same article)

Lexical definitions (all 0-1 unless noted):
  bleu      sacrebleu.sentence_bleu(hyp, [ref], smooth_method="exp") / 100   (== madlad_v1.bleu == pilot.bleu)
  rougeL    rouge_score RougeScorer(["rougeL"], use_stemmer=True) F-measure
  chrf      sacrebleu CHRF() defaults (char order 6, beta 2) / 100
  word_edit_rate  word-level Levenshtein distance / ref words (WER-style edit rate, no block shifts; can exceed
                  1). Used instead of TER, whose shift search took ~18 s on the 916-word document.
  word_retention  clipped multiset recall of lowercase \\b\\w+\\b tokens (== madlad_v1.word_retention)
  length_ratio    hyp words / ref words (whitespace words);  exact_equal  ref.strip() == hyp.strip()
Style distances: function-word cosine, punctuation L1, UPOS and dependency-label Jensen-Shannon divergence,
character 3-gram cosine, Burrows' Delta (z-scores from T0 corpus means/SDs of the 150 most frequent words),
sentence-length ratio. Information: entity recall (string presence of ref entities in hyp), number recall/additions
(multisets of normalized numbers), quantity recall (number+unit). These are HEURISTIC WARNINGS, not validated
factual errors.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter

import numpy as np
import pandas as pd

from . import paths
from .corpus import ORIGINAL
from .features import PUNCT_NAMES, UPOS, WORD

METRIC_VERSION = "pair-metrics-v2"
PAIR_KEY = ["ref_sha256", "hyp_sha256"]


# -- pair list ----------------------------------------------------------------------------------------------

def build_pairs(en: pd.DataFrame, seed: int) -> pd.DataFrame:
    """`en`: valid English texts (T0 rows have translator == "original"). One row per pair role."""
    t0 = en[en.stage == "T0"].set_index("item_id")
    rows = []
    for (t, item), part in en[en.translator != ORIGINAL].groupby(["translator", "item_id"]):
        avail = {s: r for s, r in zip(part.stage, part.itertuples(index=False))}
        chain = [t0.loc[item]] + [avail.get(f"T{k}") for k in (1, 2, 3)]
        for k in (1, 2, 3):
            hyp = chain[k]
            if hyp is None:
                continue
            rows.append(_pair(t, item, "cumulative", "T0", f"T{k}", chain[0], hyp))
            prev = chain[k - 1]
            if prev is not None:
                rows.append(_pair(t, item, "step", "T0" if k == 1 else f"T{k-1}", f"T{k}", prev, hyp))
    # baselines (translator-free)
    rng = np.random.default_rng(seed)
    items = t0.reset_index()
    perm = rng.permutation(len(items))
    for i, j0 in enumerate(perm):
        a = items.iloc[i]
        j = j0
        while items.iloc[j].article_id == a.article_id:          # deterministic: walk to the next other article
            j = (j + 1) % len(items)
        rows.append(_pair("baseline", a.item_id, "unrelated", "T0", "T0", a, items.iloc[j]))
    human = items[items.is_human].set_index("article_id")
    for a in items[~items.is_human].itertuples(index=False):
        if a.article_id in human.index:
            h = human.loc[a.article_id]
            rows.append(_pair("baseline", a.item_id, "cross_source", "T0", "T0", h, a))
    p = pd.DataFrame(rows)
    p["round"] = p.hyp_stage.str[1].astype(int)
    return p


def _pair(t, item, kind, rs, hs, ref, hyp) -> dict:
    g = (lambda r, k: getattr(r, k) if not isinstance(r, pd.Series) else r[k])
    return {"translator": t, "item_id": item, "kind": kind, "ref_stage": rs, "hyp_stage": hs,
            "ref_text_id": g(ref, "text_id"), "hyp_text_id": g(hyp, "text_id"),
            "ref_sha256": g(ref, "text_sha256"), "hyp_sha256": g(hyp, "text_sha256")}


# -- metric functions ---------------------------------------------------------------------------------------

_SCORERS = {}


def _scorers():
    if not _SCORERS:
        import sacrebleu
        from rouge_score import rouge_scorer

        _SCORERS.update(rouge=rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True), chrf=sacrebleu.CHRF(),
                        bleu=sacrebleu)
    return _SCORERS


def levenshtein(a: list, b: list) -> int:
    """Word-level edit distance (insertions, deletions, substitutions), O(len(a) * len(b))."""
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def lexical(ref: str, hyp: str) -> dict:
    s = _scorers()
    rt = Counter(w.lower() for w in WORD.findall(ref))
    ht = Counter(w.lower() for w in WORD.findall(hyp))
    return {"bleu": s["bleu"].sentence_bleu(hyp, [ref], smooth_method="exp").score / 100,
            "rougeL": s["rouge"].score(ref, hyp)["rougeL"].fmeasure,
            "chrf": s["chrf"].sentence_score(hyp, [ref]).score / 100,
            "word_edit_rate": levenshtein(ref.split(), hyp.split()) / max(len(ref.split()), 1),
            "word_retention": sum(min(c, ht[w]) for w, c in rt.items()) / max(sum(rt.values()), 1),
            "length_ratio": len(hyp.split()) / max(len(ref.split()), 1),
            "exact_equal": float(ref.strip() == hyp.strip())}


def _cos(a, b) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else np.nan


def _jsd(p, q) -> float:
    p, q = np.asarray(p, float), np.asarray(q, float)
    if p.sum() == 0 or q.sum() == 0:
        return np.nan
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2

    def kl(a, b):
        nz = a > 0
        return float((a[nz] * np.log2(a[nz] / b[nz])).sum())
    return (kl(p, m) + kl(q, m)) / 2


def _char3(t: str) -> Counter:
    t = re.sub(r"\s+", " ", t.lower())
    return Counter(t[i:i + 3] for i in range(len(t) - 2))


def style(ref: str, hyp: str, rf: pd.Series, hf: pd.Series, refl: dict) -> dict:
    fw = [f"fw_{w}" for w in refl["function_words"]]
    pu = list(PUNCT_NAMES)
    pos = [f"pos_{p}" for p in UPOS]
    dep = [c for c in rf.index if c.startswith("dep_") and c not in ("dep_dist_mean", "dep_depth_mean",
                                                                       "dep_depth_max")]
    mfw = [f"mfw_{w}" for w in refl["mfw"]]
    mu, sd = np.array(refl["mfw_mean"]), np.array(refl["mfw_sd"])
    sd = np.where(sd > 0, sd, np.nan)
    zr, zh = (rf[mfw].to_numpy(float) - mu) / sd, (hf[mfw].to_numpy(float) - mu) / sd
    c3r, c3h = _char3(ref), _char3(hyp)
    keys = sorted(set(c3r) | set(c3h))
    return {"fw_cosine": _cos(rf[fw].to_numpy(float), hf[fw].to_numpy(float)),
            "punct_l1": float(np.abs(rf[pu].to_numpy(float) - hf[pu].to_numpy(float)).sum()),
            "pos_jsd": _jsd(rf[pos].to_numpy(float), hf[pos].to_numpy(float)),
            "dep_jsd": _jsd(rf[dep].to_numpy(float), hf[dep].to_numpy(float)),
            "char3_cosine": _cos(np.array([c3r[k] for k in keys], float), np.array([c3h[k] for k in keys], float)),
            "burrows_delta": float(np.nanmean(np.abs(zr - zh))),
            "sent_len_ratio": hf["sent_len_mean"] / rf["sent_len_mean"] if rf["sent_len_mean"] else np.nan,
            "sentence_count_ratio": hf["n_sentences"] / max(rf["n_sentences"], 1)}


def _present(entity: str, text_lower: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(entity) + r"(?!\w)", text_lower) is not None


def information(ref_f: pd.Series, hyp: str, hyp_f: pd.Series) -> dict:
    ents = [e for e, _ in json.loads(ref_f["entities_json"])]
    low = hyp.lower()
    lost_e = [e for e in ents if not _present(e, low)]
    rn, hn = Counter(json.loads(ref_f["numbers_json"])), Counter(json.loads(hyp_f["numbers_json"]))
    rq, hq = Counter(json.loads(ref_f["quantities_json"])), Counter(json.loads(hyp_f["quantities_json"]))
    kept_n = sum(min(c, hn[k]) for k, c in rn.items())
    kept_q = sum(min(c, hq[k]) for k, c in rq.items())
    return {"entities_ref": len(ents), "entities_lost": len(lost_e),
            "entity_recall": 1 - len(lost_e) / len(ents) if ents else np.nan,
            "numbers_ref": sum(rn.values()), "numbers_lost": sum(rn.values()) - kept_n,
            "numbers_added": sum(hn.values()) - kept_n,
            "number_recall": kept_n / sum(rn.values()) if rn else np.nan,
            "quantities_ref": sum(rq.values()),
            "quantity_recall": kept_q / sum(rq.values()) if rq else np.nan,
            "lost_entities_json": json.dumps(lost_e),
            "lost_numbers_json": json.dumps(sorted((rn - hn).elements())),
            "added_numbers_json": json.dumps(sorted((hn - rn).elements()))}


LOCAL_ONLY = ["lost_entities_json", "lost_numbers_json", "added_numbers_json"]


def fingerprint(feature_fp: str) -> str:
    import rouge_score
    import sacrebleu

    blob = json.dumps({"v": METRIC_VERSION, "features": feature_fp, "sacrebleu": sacrebleu.__version__,
                       "rouge_score": getattr(rouge_score, "__version__", "0.1.2")}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _one(args):
    ref, hyp, rf, hf, refl = args
    return {**lexical(ref, hyp), **style(ref, hyp, rf, hf, refl), **information(rf, hyp, hf)}


def compute_cpu(pairs: pd.DataFrame, text_of: dict, feats: pd.DataFrame, refl: dict, feature_fp: str,
                workers: int = 4, log=print) -> pd.DataFrame:
    """Lexical + style + information metrics for each unique (ref, hyp), resuming from the cache."""
    from concurrent.futures import ProcessPoolExecutor

    fp = fingerprint(feature_fp)
    cache = paths.cache(f"pair_cpu__{fp}.parquet")
    done = pd.read_parquet(cache) if cache.exists() else pd.DataFrame(columns=PAIR_KEY)
    uniq = pairs.drop_duplicates(PAIR_KEY)[PAIR_KEY]
    have = set(map(tuple, done[PAIR_KEY].to_numpy()))
    todo = uniq[[k not in have for k in map(tuple, uniq.to_numpy())]]
    log(f"pair CPU metrics [{fp}]: {len(uniq)} unique pairs, {len(uniq) - len(todo)} cached, {len(todo)} to score")
    F = feats.set_index("text_sha256")
    for start in range(0, len(todo), 400):
        part = todo.iloc[start:start + 400]
        args = [(text_of[r], text_of[h], F.loc[r], F.loc[h], refl) for r, h in part.to_numpy()]
        with ProcessPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(_one, args, chunksize=8))
        new = pd.concat([part.reset_index(drop=True), pd.DataFrame(res)], axis=1)
        done = pd.concat([done, new], ignore_index=True) if len(done) else new
        tmp = cache.with_suffix(".tmp")
        done.to_parquet(tmp, index=False)
        tmp.replace(cache)
        log(f"  scored {min(start + 400, len(todo))}/{len(todo)}")
    return done, fp
