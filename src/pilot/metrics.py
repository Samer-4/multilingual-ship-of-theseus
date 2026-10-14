"""Metrics for the Ship of Theseus corpus analysis (docs/sot_analysis_plan.md).

Lexical ("planks"): BLEU, ROUGE-1/2/L, content-word retention.
Semantic ("hull"): BERTScore (roberta-large, layer 17, baseline-rescaled) for texts of any length,
                   plus a chunked sentence-embedding cosine as a check.
Style: dense stylometric vectors (surface, punctuation, function words, POS, dependency depth) and
       character n-gram TF-IDF.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

DATASETS = ["cmv", "eli5", "sci_gen", "tldr", "wp", "xsum", "yelp"]
WORD = re.compile(r"[A-Za-z]+(?:'[a-z]+)?")


def best_device(device: str = "auto") -> str:
    """'auto' -> cuda (e.g. Explorer GPU nodes), else mps (Apple silicon), else cpu."""
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


# -- corpus -------------------------------------------------------------------------------------

def load_corpus(sot_dir: Path, datasets=DATASETS) -> pd.DataFrame:
    """Long table of every text: dataset, source, key, variant, round (0 = original), text."""
    frames = []
    for ds in datasets:
        d = pd.read_csv(Path(sot_dir) / f"paraphrased_datasets/{ds}_paraphrased.csv")
        d["dataset"] = ds
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["text"] = d.text.astype(str)
    d["round"] = np.where(d.version_name == "original", 0, d.version_name.str.count("_") + 1)
    d["variant"] = np.where(d.version_name == "original", "original", d.version_name.str.split("_").str[0])
    return d[["dataset", "source", "key", "variant", "round", "version_name", "text"]]


def load_reference(sot_dir: Path, datasets=DATASETS) -> pd.DataFrame:
    frames = []
    for ds in datasets:
        d = pd.read_csv(Path(sot_dir) / f"train_datasets/{ds}_train.csv")
        d["dataset"] = ds
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    d["text"] = d.text.astype(str)
    return d[["dataset", "source", "key", "text"]]


# -- lexical ------------------------------------------------------------------------------------

_ROUGE = None


def lexical_scores(ref: str, hyp: str, stopwords: frozenset) -> dict:
    """BLEU (sentence, exp smoothing), ROUGE-1/2/L F1 (stemmed), content-word type retention; all 0-1."""
    global _ROUGE
    import sacrebleu
    from rouge_score import rouge_scorer

    if _ROUGE is None:
        _ROUGE = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r = _ROUGE.score(ref, hyp)
    ref_words = {w.lower() for w in WORD.findall(ref)} - stopwords
    hyp_words = {w.lower() for w in WORD.findall(hyp)}
    return {
        "bleu": sacrebleu.sentence_bleu(hyp, [ref], smooth_method="exp").score / 100,
        "rouge1": r["rouge1"].fmeasure, "rouge2": r["rouge2"].fmeasure, "rougeL": r["rougeL"].fmeasure,
        "word_retention": len(ref_words & hyp_words) / len(ref_words) if ref_words else np.nan,
    }


# -- BERTScore for long texts -------------------------------------------------------------------

class LongBertScore:
    """BERTScore as in bert_score 0.3.13 (roberta-large, layer 17, no idf, baseline rescaling), but
    texts longer than the 512-token window are embedded in overlapping windows instead of truncated.
    Each token's embedding comes from the window where it sits in the core region, so it always has
    at least `margin` tokens of context on each side (except at the text's ends).
    """

    def __init__(self, device: str = "auto", model_type: str = "roberta-large", layer: int = 17,
                 window: int = 510, step: int = 384):
        import bert_score
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device = best_device(device)
        self.tok = AutoTokenizer.from_pretrained(model_type)
        model = AutoModel.from_pretrained(model_type)
        model.encoder.layer = model.encoder.layer[:layer]  # bert_score truncates the encoder at `layer`
        self.model = model.to(device).eval()
        self.window, self.step, self.margin = window, step, (window - step) // 2
        base = Path(bert_score.__file__).parent / f"rescale_baseline/en/{model_type}.tsv"
        row = pd.read_csv(base).set_index("LAYER").loc[layer]
        self.baseline = {"P": row["P"], "R": row["R"], "F": row["F"]}
        self.truncation_avoided = 0

    def _windows(self, n: int):
        if n <= self.window:
            return [(0, n, 0, n)]
        out = []
        for c in range(0, n, self.step):
            d = min(c + self.step, n)
            a, b = max(0, c - self.margin), min(n, d + self.margin)
            out.append((a, b, c, d))
        return out

    def embed(self, texts: list[str], batch_size: int = 16) -> list:
        """Unit-normalized layer embeddings per token (special tokens excluded), one tensor per text."""
        torch = self.torch
        pieces = []  # (text index, window ids, core slice within window)
        # Same tokenization as bert_score.utils.sent_encode for RoBERTa: stripped, with a prefix space.
        all_ids = [self.tok(t.strip(), add_special_tokens=False, add_prefix_space=True)["input_ids"] for t in texts]
        for i, ids in enumerate(all_ids):
            wins = self._windows(len(ids))
            if len(wins) > 1:
                self.truncation_avoided += 1
            for a, b, c, d in wins:
                pieces.append((i, ids[a:b], c - a, d - a))
        parts = {i: [] for i in range(len(texts))}
        cls, sep, pad = self.tok.cls_token_id, self.tok.sep_token_id, self.tok.pad_token_id
        for k in range(0, len(pieces), batch_size):
            batch = pieces[k:k + batch_size]
            seqs = [[cls] + w + [sep] for _, w, _, _ in batch]
            L = max(len(s) for s in seqs)
            ids = torch.tensor([s + [pad] * (L - len(s)) for s in seqs], device=self.device)
            mask = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs], device=self.device)
            with torch.no_grad():
                h = self.model(input_ids=ids, attention_mask=mask).last_hidden_state
            lens = mask.sum(dim=1)
            for (i, _, c, d), hi, n in zip(batch, h, lens):
                parts[i].append((hi[0], hi[1 + c:1 + d], hi[n - 1]))  # (<s>, core tokens, </s>)
        out = []
        for i in range(len(texts)):
            # Rows: <s> of the first window, every token, </s> of the last window. bert_score keeps the
            # special tokens as match targets (zero weight only on the averaging side), so do we.
            first, last = parts[i][0][0], parts[i][-1][2]
            e = torch.cat([first[None], *[core for _, core, _ in parts[i]], last[None]])
            out.append(torch.nn.functional.normalize(e, dim=-1))
        return out

    def score(self, cand, ref) -> dict:
        """Greedy-matching P/R/F between two embedded texts, baseline-rescaled."""
        sim = cand @ ref.T  # rows/cols 0 and -1 are the special tokens: targets, not averaged
        p = sim[1:-1].max(dim=1).values.mean().item()
        r = sim[:, 1:-1].max(dim=0).values.mean().item()
        f = 2 * p * r / (p + r) if p + r else 0.0
        b = self.baseline
        return {"bertscore_p": (p - b["P"]) / (1 - b["P"]), "bertscore_r": (r - b["R"]) / (1 - b["R"]),
                "bertscore_f": (f - b["F"]) / (1 - b["F"])}


class ChunkedSentenceEmbedder:
    """Sentence-embedding cosine for long texts: mean of normalized chunk embeddings (~200 words each)."""

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", device: str = "auto",
                 chunk_words: int = 200):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=best_device(device))
        self.chunk_words = chunk_words

    def embed(self, texts: list[str]) -> np.ndarray:
        chunks, owner = [], []
        for i, t in enumerate(texts):
            words = t.split() or [""]
            for k in range(0, len(words), self.chunk_words):
                chunks.append(" ".join(words[k:k + self.chunk_words]))
                owner.append(i)
        e = self.model.encode(chunks, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        owner = np.array(owner)
        out = np.stack([e[owner == i].mean(axis=0) for i in range(len(texts))])
        return out / np.linalg.norm(out, axis=1, keepdims=True)


# -- style --------------------------------------------------------------------------------------

PUNCT = [",", ".", ";", ":", "!", "?", '"', "'", "(", "-"]


def mattr(tokens: list[str], window: int = 50) -> float:
    """Moving-average type-token ratio (length-robust lexical diversity)."""
    if len(tokens) < 2:
        return np.nan
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    return float(np.mean([len(set(tokens[i:i + window])) / window for i in range(len(tokens) - window + 1)]))


def _depths(doc) -> np.ndarray:
    """Dependency depth of every token in O(n): one pass over head indices, memoized.
    (Walking token.head per token creates a spaCy Token object at every step, which was the bottleneck.)"""
    heads = np.array([t.head.i for t in doc], dtype=np.int64)
    depth = np.full(len(heads), -1, dtype=np.int64)
    for i in range(len(heads)):
        path, j = [], i
        while depth[j] < 0 and heads[j] != j:
            path.append(j)
            j = heads[j]
        base = 0 if heads[j] == j and depth[j] < 0 else depth[j]
        if depth[j] < 0:
            depth[j] = 0
        for k in reversed(path):
            base += 1
            depth[k] = base
    return depth


def style_features(doc, function_words: list[str], pos_tags: list[str]) -> dict:
    """Dense stylometric features of one spaCy doc (all rates per token, so length-normalized)."""
    toks = [t for t in doc if not t.is_space]
    n = max(len(toks), 1)
    words = [t.text.lower() for t in toks if t.is_alpha]
    sents = [s for s in doc.sents if any(not t.is_space for t in s)]
    sent_lens = [sum(1 for t in s if not t.is_space) for s in sents] or [0]
    feats = {
        "sent_len_mean": float(np.mean(sent_lens)), "sent_len_sd": float(np.std(sent_lens)),
        "word_len_mean": float(np.mean([len(w) for w in words])) if words else 0.0,
        "mattr": mattr(words), "cap_rate": sum(t.text[:1].isupper() for t in toks if t.is_alpha) / max(len(words), 1),
        "dep_depth_mean": float(np.mean(_depths(doc)[[t.i for t in toks]])) if toks else 0.0,
    }
    text = doc.text
    for p in PUNCT:
        feats[f"punct_{p}"] = text.count(p) / n
    counts = pd.Series(words).value_counts() if words else pd.Series(dtype=float)
    for w in function_words:
        feats[f"fw_{w}"] = counts.get(w, 0) / n
    pos = pd.Series([t.pos_ for t in toks]).value_counts() if toks else pd.Series(dtype=float)
    for p in pos_tags:
        feats[f"pos_{p}"] = pos.get(p, 0) / n
    return feats


# -- statistics ---------------------------------------------------------------------------------

def bootstrap_mean_ci(df: pd.DataFrame, group_cols: list[str], value: str, cluster: str = "key",
                      n_boot: int = 1000, seed: int = 6120) -> pd.DataFrame:
    """Mean and 95% percentile CI per group, resampling clusters (keys) with replacement."""
    rng = np.random.default_rng(seed)
    rows = []
    for g, part in df.groupby(group_cols, observed=True):
        per_cluster = part.groupby(cluster)[value].agg(["sum", "count"])
        s, c = per_cluster["sum"].to_numpy(), per_cluster["count"].to_numpy()
        idx = rng.integers(0, len(s), size=(n_boot, len(s)))
        boots = s[idx].sum(axis=1) / c[idx].sum(axis=1)
        g = g if isinstance(g, tuple) else (g,)
        rows.append({**dict(zip(group_cols, g)), "mean": s.sum() / c.sum(),
                     "ci_low": np.percentile(boots, 2.5), "ci_high": np.percentile(boots, 97.5), "n": int(c.sum())})
    return pd.DataFrame(rows)


# -- semantic scoring over chains (resumable) ---------------------------------------------------

def chain_jobs(sample: pd.DataFrame, texts: pd.Series, variants: list[str], id_cols: list[str]):
    """One job per original: [T0, every Tn of every variant] plus the (ref, hyp, meta) index pairs to score.
    Embedding each text once and scoring all its pairs avoids re-embedding T0 and shared rounds."""
    for o in sample.itertuples(index=False):
        base = tuple(getattr(o, c) for c in id_cols)
        seq = [(v, n) for v in variants for n in (1, 2, 3) if (*base, v, n) in texts.index]
        tx = [texts[(*base, "original", 0)]] + [texts[(*base, v, n)] for v, n in seq]
        idx = {("original", 0): 0, **{s: i + 1 for i, s in enumerate(seq)}}
        prs = []
        for v, n in seq:
            prs.append((0, idx[(v, n)], {"variant": v, "round": n, "kind": "cumulative"}))
            if n > 1 and (v, n - 1) in idx:
                prs.append((idx[(v, n - 1)], idx[(v, n)], {"variant": v, "round": n, "kind": "step"}))
        yield dict(zip(id_cols, base)), tx, prs


def score_semantic(jobs, cache_path: Path, id_cols: list[str], device: str = "auto", flush_every: int = 50,
                   log_every: int = 200) -> pd.DataFrame:
    """BERTScore (P/R/F) and chunked sentence-embedding cosine for every pair; appends to a parquet cache
    every `flush_every` jobs and skips jobs already cached, so an interrupted run resumes."""
    import time

    cache_path = Path(cache_path)
    done = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame()
    done_ids = set(map(tuple, done[id_cols].drop_duplicates().to_numpy())) if len(done) else set()
    bs, sb = LongBertScore(device), ChunkedSentenceEmbedder(device=device)
    rows, n_jobs, started = [], 0, time.time()

    def flush():
        nonlocal done, rows
        if rows:
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
            done.to_parquet(cache_path, index=False)
            rows = []

    for meta, tx, prs in jobs:
        if tuple(meta[c] for c in id_cols) in done_ids:
            continue
        be, se = bs.embed(tx), sb.embed(tx)
        for r, h, pm in prs:
            rows.append({**meta, **pm, **bs.score(be[h], be[r]), "sbert_cos": float(se[h] @ se[r])})
        n_jobs += 1
        if n_jobs % flush_every == 0:
            flush()
        if n_jobs % log_every == 0:
            print(f"{n_jobs} jobs, {time.time() - started:.0f}s", flush=True)
    flush()
    print(f"semantic scoring done: {n_jobs} new jobs, {len(done):,} rows, {time.time() - started:.0f}s; "
          f"long texts windowed: {bs.truncation_avoided}", flush=True)
    return done


# -- style feature extraction (shared by the notebook and scripts/style_features.py) -------------

UPOS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN",
        "PUNCT", "SCONJ", "SYM", "VERB", "X"]


def text_hash(text: str) -> str:
    import hashlib

    return hashlib.sha1(text.encode()).hexdigest()


def reference_subset(reference: pd.DataFrame, n_per_dataset: int, seed: int) -> pd.DataFrame:
    """The reference texts that fix the function-word list, the z-scoring and the char n-gram vocabulary."""
    return (reference.groupby("dataset", group_keys=False)
            .apply(lambda d: d.sample(n_per_dataset, random_state=seed)).reset_index(drop=True))


def top_function_words(texts, n: int = 100) -> list[str]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    words = pd.Series([w.lower() for t in texts for w in t.split()])
    return list(words[words.isin(ENGLISH_STOP_WORDS)].value_counts().index[:n])


def extract_style(texts: list[str], function_words: list[str], n_process: int = 1) -> pd.DataFrame:
    """spaCy parse + style_features for each text; returns one row per text with its `text_hash`.
    Use n_process > 1 only from a script with a __main__ guard (it deadlocks inside Jupyter on macOS)."""
    import spacy

    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    nlp.max_length = 2_000_000
    rows = [style_features(doc, function_words, UPOS) for doc in nlp.pipe(texts, batch_size=64, n_process=n_process)]
    return pd.DataFrame(rows).assign(text_hash=[text_hash(t) for t in texts])


# -- pairs and chunked lexical scoring (shared by the notebook and scripts/score_lexical.py) -----

PAIR_KEYS = ["dataset", "source", "key", "variant", "round", "kind"]


def build_pairs(corpus: pd.DataFrame, sample: pd.DataFrame, variants: list[str]) -> pd.DataFrame:
    """Cumulative (T0 vs Tn) and step (Tn-1 vs Tn) pairs for the originals in `sample`, built with merges."""
    ids = ["dataset", "source", "key"]
    t0 = corpus[corpus["round"] == 0].set_index(ids).text.rename("t0")
    c = (corpus[(corpus["round"] > 0) & corpus.variant.isin(variants)].merge(sample[ids], on=ids)
         .join(t0, on=ids)[ids + ["variant", "round", "t0", "text"]])
    cumulative = c.rename(columns={"t0": "ref", "text": "hyp"}).assign(kind="cumulative")
    prev = c[ids + ["variant", "round", "text"]].assign(round=lambda d: d["round"] + 1).rename(columns={"text": "ref"})
    step = (c[c["round"] > 1].merge(prev, on=ids + ["variant", "round"])[ids + ["variant", "round", "ref", "text"]]
            .rename(columns={"text": "hyp"}).assign(kind="step"))
    pairs = pd.concat([cumulative, step], ignore_index=True)[PAIR_KEYS + ["ref", "hyp"]]
    pairs = pairs.sort_values(PAIR_KEYS).reset_index(drop=True)
    assert not pairs.duplicated(PAIR_KEYS).any()
    assert pairs[["ref", "hyp"]].notna().all().all(), "an original in `sample` has no T0 text in the corpus"
    return pairs


def score_lexical_chunked(pairs: pd.DataFrame, out_path: Path, workers: int = 6, chunk: int = 20_000) -> pd.DataFrame:
    """Lexical scores in bounded-memory chunks (pool.map pickles every task up front, so feeding it 500k+
    text pairs at once exhausts memory). Each chunk is saved to a parts folder, so the run resumes."""
    import time
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial

    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    out_path = Path(out_path)
    parts_dir = out_path.with_suffix(".parts")
    parts_dir.mkdir(exist_ok=True)
    fn = partial(lexical_scores, stopwords=frozenset(ENGLISH_STOP_WORDS))
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, start in enumerate(range(0, len(pairs), chunk)):
            part = parts_dir / f"part_{i:04d}.parquet"
            if part.exists():
                continue
            frame = pairs.iloc[start:start + chunk]
            rows = list(pool.map(fn, frame.ref, frame.hyp, chunksize=200))
            pd.concat([frame[PAIR_KEYS].reset_index(drop=True), pd.DataFrame(rows)], axis=1).to_parquet(part, index=False)
            print(f"lexical chunk {i + 1}/{-(-len(pairs) // chunk)} done, {time.time() - started:.0f}s", flush=True)
    lex = pd.concat([pd.read_parquet(p) for p in sorted(parts_dir.glob("part_*.parquet"))], ignore_index=True)
    assert len(lex) == len(pairs), "lexical parts don't cover all pairs"
    lex.to_parquet(out_path, index=False)
    return lex
