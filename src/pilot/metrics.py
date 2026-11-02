"""Metrics for the Ship of Theseus analyses, following the team metric standard (docs/metric_standard.md).

Lexical ("planks"): BLEU, ROUGE-1/2/L, content-word retention (stop words excluded).
Semantic ("hull"): BERTScore (roberta-large, layer 17, baseline-rescaled) for texts of any length, and MPNet
                   cosine (sentence-packed 384-token chunks, token-count-weighted mean; Samer's method).
Style: the union of both team pipelines, combined into one score by feature family (STYLE_FAMILIES).
Every score is reported on the retained scale against unrelated T0 pairs of the same dataset (`retained`).
"""

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

DATASETS = ["cmv", "eli5", "sci_gen", "tldr", "wp", "xsum", "yelp"]
WORD = re.compile(r"[A-Za-z]+(?:'[a-z]+)?")


def half_precision(device: str, dtype: str) -> bool:
    """fp16 weights only on CUDA and only when asked for; fp32 is the reference (bit-comparable across machines)."""
    if dtype not in ("fp32", "fp16"):
        raise ValueError(f"dtype must be 'fp32' or 'fp16', not {dtype!r}")
    return dtype == "fp16" and device == "cuda"


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
                 window: int = 510, step: int = 384, dtype: str = "fp32", batch_size: int = 16):
        import bert_score
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device = best_device(device)
        self.tok = AutoTokenizer.from_pretrained(model_type)
        model = AutoModel.from_pretrained(model_type)
        model.encoder.layer = model.encoder.layer[:layer]  # bert_score truncates the encoder at `layer`
        self.half = half_precision(device, dtype)
        self.model = (model.half() if self.half else model).to(device).eval()
        self.batch_size = batch_size
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

    def embed(self, texts: list[str], batch_size: int | None = None) -> list:
        """Unit-normalized layer embeddings per token (special tokens excluded), one fp32 tensor per text.
        Windows are batched in length order, so little compute goes to padding."""
        torch = self.torch
        batch_size = batch_size or self.batch_size
        pieces = []  # (text index, window index, window ids, core slice within window)
        # Same tokenization as bert_score.utils.sent_encode for RoBERTa: stripped, with a prefix space.
        all_ids = [self.tok(t.strip(), add_special_tokens=False, add_prefix_space=True)["input_ids"] for t in texts]
        for i, ids in enumerate(all_ids):
            wins = self._windows(len(ids))
            if len(wins) > 1:
                self.truncation_avoided += 1
            for j, (a, b, c, d) in enumerate(wins):
                pieces.append((i, j, ids[a:b], c - a, d - a))
        pieces.sort(key=lambda x: len(x[2]))
        parts = {i: {} for i in range(len(texts))}
        cls, sep, pad = self.tok.cls_token_id, self.tok.sep_token_id, self.tok.pad_token_id
        for k in range(0, len(pieces), batch_size):
            batch = pieces[k:k + batch_size]
            seqs = [[cls] + w + [sep] for _, _, w, _, _ in batch]
            L = max(len(s) for s in seqs)
            ids = torch.tensor([s + [pad] * (L - len(s)) for s in seqs], device=self.device)
            mask = torch.tensor([[1] * len(s) + [0] * (L - len(s)) for s in seqs], device=self.device)
            with torch.no_grad():
                h = self.model(input_ids=ids, attention_mask=mask).last_hidden_state.float()
            lens = mask.sum(dim=1)
            for (i, j, _, c, d), hi, n in zip(batch, h, lens):
                parts[i][j] = (hi[0], hi[1 + c:1 + d], hi[n - 1])  # (<s>, core tokens, </s>)
        out = []
        for i in range(len(texts)):
            # Rows: <s> of the first window, every token, </s> of the last window. bert_score keeps the
            # special tokens as match targets (zero weight only on the averaging side), so do we.
            wins = [parts[i][j] for j in sorted(parts[i])]
            e = torch.cat([wins[0][0][None], *[core for _, core, _ in wins], wins[-1][2][None]])
            out.append(torch.nn.functional.normalize(e, dim=-1))
        return out

    def score(self, cand, ref) -> dict:
        """Greedy-matching P/R/F between two embedded texts, baseline-rescaled."""
        sim = cand @ ref.T  # rows/cols 0 and -1 are the special tokens: targets, not averaged
        p = sim[1:-1].max(dim=1).values.mean().item()
        r = sim[:, 1:-1].max(dim=0).values.mean().item()
        f = 2 * p * r / (p + r) if p + r else 0.0
        b = self.baseline
        return {"bertscore_p_rescaled": (p - b["P"]) / (1 - b["P"]),
                "bertscore_r_rescaled": (r - b["R"]) / (1 - b["R"]),
                "bertscore_f1_rescaled": (f - b["F"]) / (1 - b["F"])}


SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


class SentenceChunkEmbedder:
    """Document cosine for texts of any length (same method as Samer's sot_research.neural.ChunkEmbedder).
    Each text is split into sentences, packed greedily into chunks that fit the encoder window, each chunk
    mean-pooled and L2-normalized; the document vector is the token-count-weighted mean of its chunk vectors,
    re-normalized. A text that fits one window gets the standard single-pass embedding. Nothing is truncated."""

    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", max_tokens: int = 384,
                 device: str = "auto", batch_size: int = 16, dtype: str = "fp32", prefix: str = ""):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.device = best_device(device)
        self.half = half_precision(self.device, dtype)
        model = AutoModel.from_pretrained(model_name)
        self.model = (model.half() if self.half else model).to(self.device).eval()
        self.max_tokens, self.batch_size, self.prefix = max_tokens, batch_size, prefix
        self.chunked_texts = 0
        self.split_words = 0  # over-long single words split by characters (see chunks)

    def _len(self, s: str) -> int:
        return len(self.tok(self.prefix + s, add_special_tokens=True)["input_ids"])

    def chunks(self, text: str) -> list[str]:
        def fit(parts):
            s = " ".join(parts)
            if self._len(s) <= self.max_tokens:
                return [s]
            if len(parts) > 1:
                h = len(parts) // 2
                return fit(parts[:h]) + fit(parts[h:])
            words = s.split()
            if len(words) <= 1:
                # One whitespace-free run longer than the window (repetition loops such as "f-f-f-...",
                # ");););..."): halve it by characters. Nothing is dropped; Samer's code raises here instead,
                # and gives identical results on every text that doesn't reach this branch.
                self.split_words += 1
                h = len(s) // 2
                return fit([s[:h]]) + fit([s[h:]])
            h = len(words) // 2
            return fit([" ".join(words[:h])]) + fit([" ".join(words[h:])])

        sents = [x for x in SENT_SPLIT.split(text.strip()) if x] or [text.strip()]
        if self._len(" ".join(sents)) <= self.max_tokens:
            return [" ".join(sents)]
        out, cur = [], []
        for s in sents:
            if cur and self._len(" ".join(cur + [s])) > self.max_tokens:
                out += fit(cur)
                cur = []
            cur.append(s)
        return out + fit(cur) if cur else out

    def embed(self, texts: list[str]) -> np.ndarray:
        torch = self.torch
        chunks, owner = [], []
        for i, t in enumerate(texts):
            c = self.chunks(t)
            self.chunked_texts += len(c) > 1
            chunks += c
            owner += [i] * len(c)
        order = sorted(range(len(chunks)), key=lambda k: len(chunks[k]))  # length-sorted batches: less padding
        v, n = np.zeros((len(chunks), self.model.config.hidden_size)), np.zeros(len(chunks))
        for k in range(0, len(order), self.batch_size):
            idx = order[k:k + self.batch_size]
            b = self.tok([self.prefix + chunks[i] for i in idx], padding=True, truncation=False,
                         return_tensors="pt").to(self.device)
            if b["input_ids"].shape[1] > self.max_tokens:
                raise ValueError("chunk exceeds the window")  # never truncate silently
            with torch.no_grad():
                h = self.model(**b).last_hidden_state.float()
            m = b["attention_mask"].unsqueeze(-1).float()
            v[idx] = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1).cpu().numpy()
            n[idx] = b["attention_mask"].sum(1).cpu().numpy()
        owner = np.array(owner)
        out = np.stack([(v[owner == i] * n[owner == i, None]).sum(0) for i in range(len(texts))])
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


def style_features(doc, function_words: list[str], pos_tags: list[str], dep_labels: list[str] = (),
                   mfw: list[str] = ()) -> dict:
    """Dense stylometric features of one spaCy doc (all rates per token, so length-normalized).
    deplab_* (dependency-label rates) and mfw_* (most-frequent-word rates, for Burrows' Delta) feed the
    distance metrics ported from Samer's pipeline."""
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
    dep = pd.Series([t.dep_ for t in toks]).value_counts() if toks else pd.Series(dtype=float)
    for d in dep_labels:
        feats[f"deplab_{d}"] = dep.get(d, 0) / n
    for w in mfw:
        feats[f"mfw_{w}"] = counts.get(w, 0) / n
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
                   log_every: int = 200, dtype: str = "fp32", jobs_per_batch: int = 1, batch_size: int = 16) -> pd.DataFrame:
    """Rescaled BERTScore (P/R/F1) and MPNet cosine for every pair; appends to a parquet cache
    every `flush_every` jobs and skips jobs already cached, so an interrupted run resumes.
    On a large GPU, raise `jobs_per_batch` (texts of many jobs embedded together) and `batch_size`;
    `dtype="fp16"` halves the weights on CUDA (validate against fp32 first)."""
    import time

    cache_path = Path(cache_path)
    done = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame()
    done_ids = set(map(tuple, done[id_cols].drop_duplicates().to_numpy())) if len(done) else set()
    bs = LongBertScore(device, dtype=dtype, batch_size=batch_size)
    sb = SentenceChunkEmbedder(device=device, dtype=dtype, batch_size=batch_size)
    rows, n_jobs, started, group = [], 0, time.time(), []

    def flush():
        nonlocal done, rows
        if rows:
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
            tmp = cache_path.with_suffix(".tmp.parquet")
            done.to_parquet(tmp, index=False)
            tmp.replace(cache_path)  # atomic: a disconnect mid-write never corrupts the cache
            rows = []

    def run(group):
        nonlocal n_jobs
        texts = [t for _, tx, _ in group for t in tx]
        be, se = bs.embed(texts), sb.embed(texts)
        off = 0
        for meta, tx, prs in group:
            for r, h, pm in prs:
                rows.append({**meta, **pm, **bs.score(be[off + h], be[off + r]),
                             "mpnet_cosine": float(se[off + h] @ se[off + r])})
            off += len(tx)
            n_jobs += 1
            if n_jobs % flush_every == 0:
                flush()
            if n_jobs % log_every == 0:
                print(f"{n_jobs} jobs, {time.time() - started:.0f}s", flush=True)

    for job in jobs:
        if tuple(job[0][c] for c in id_cols) in done_ids:
            continue
        group.append(job)
        if len(group) >= jobs_per_batch:
            run(group)
            group = []
    if group:
        run(group)
    flush()
    print(f"semantic scoring done: {n_jobs} new jobs, {len(done):,} rows, {time.time() - started:.0f}s; "
          f"long texts windowed (BERTScore): {bs.truncation_avoided}, chunked (MPNet): {sb.chunked_texts}, over-long words split: {sb.split_words}", flush=True)
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


def top_words(texts, n: int = 150) -> list[str]:
    """The n most frequent lowercase alphabetic words (Burrows' Delta vocabulary), fitted on reference texts."""
    words = pd.Series([w.lower() for t in texts for w in WORD.findall(t)])
    return list(words.value_counts().index[:n])


def top_function_words(texts, n: int = 100) -> list[str]:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    words = pd.Series([w.lower() for t in texts for w in t.split()])
    return list(words[words.isin(ENGLISH_STOP_WORDS)].value_counts().index[:n])


def extract_style(texts: list[str], function_words: list[str], mfw: list[str], n_process: int = 1) -> pd.DataFrame:
    """spaCy parse + style_features for each text; returns one row per text with its `text_hash`.
    Use n_process > 1 only from a script with a __main__ guard (it deadlocks inside Jupyter on macOS)."""
    import spacy

    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    nlp.max_length = 2_000_000
    dep_labels = sorted(nlp.get_pipe("parser").labels)
    rows = [style_features(doc, function_words, UPOS, dep_labels, mfw)
            for doc in nlp.pipe(texts, batch_size=64, n_process=n_process)]
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


def score_lexical_chunked(pairs: pd.DataFrame, out_path: Path, workers: int = 6, chunk: int = 20_000,
                          fn=None) -> pd.DataFrame:
    """Lexical scores in bounded-memory chunks (pool.map pickles every task up front, so feeding it 500k+
    text pairs at once exhausts memory). Each chunk is saved to a parts folder, so the run resumes."""
    import time
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial

    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    out_path = Path(out_path)
    parts_dir = out_path.with_suffix(".parts")
    parts_dir.mkdir(exist_ok=True)
    fn = fn or partial(lexical_scores, stopwords=frozenset(ENGLISH_STOP_WORDS))
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


# -- combined style score (union of both team pipelines) -----------------------------------------

# Components grouped by what they measure. The combined `style` is the mean of the family scores, each
# the mean of its components on the retained scale, so overlapping measures (e.g. two function-word
# metrics) don't get double weight and 100 function-word dimensions can't drown out 6 surface features.
STYLE_FAMILIES = {
    "surface": ["style_surface"],
    "punctuation": ["style_punctuation", "punct_l1"],
    "function_words": ["style_function_words", "fw_cosine", "burrows_delta"],
    "syntax": ["style_syntax_pos", "pos_jsd", "dep_jsd"],
    "character": ["style_charngram", "char3_cosine"],
}
STYLE_COMPONENTS = [c for cs in STYLE_FAMILIES.values() for c in cs]
DISTANCES = {"punct_l1", "pos_jsd", "dep_jsd", "burrows_delta", "word_edit_rate"}  # 0 = identical; others: 1 = identical
SURFACE = ["sent_len_mean", "sent_len_sd", "word_len_mean", "mattr", "cap_rate", "dep_depth_mean"]


def _rowcos(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    nx, ny = np.linalg.norm(x, axis=1), np.linalg.norm(y, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((nx > 0) & (ny > 0), (x * y).sum(1) / (nx * ny), np.nan)


def _rowjsd(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Jensen-Shannon divergence (log2, 0-1) per row; NaN where either distribution is empty."""
    sp, sq = p.sum(1, keepdims=True), q.sum(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        p, q = p / sp, q / sq
        m = (p + q) / 2
        kl = lambda a: np.where(a > 0, a * np.log2(a / m), 0).sum(1)
        out = (kl(p) + kl(q)) / 2
    return np.where((sp[:, 0] > 0) & (sq[:, 0] > 0), out, np.nan)


class StyleScorer:
    """Per-pair style components. Everything that needs fitting (z-scores, Burrows' Delta means/SDs, the char
    n-gram TF-IDF vocabulary) is fitted on held-out reference texts, so no rewritten output shapes the measure.

    Block cosines (z-scored features): style_surface, style_punctuation, style_function_words, style_syntax_pos.
    Char n-grams: style_charngram (char_wb 2-4 TF-IDF cosine), char3_cosine (raw char 3-gram cosine, hashed).
    Distances: punct_l1 (L1 of punctuation rates), pos_jsd / dep_jsd (Jensen-Shannon divergence of UPOS /
    dependency-label distributions), burrows_delta (mean |z| difference over the reference's most frequent words).
    """

    def __init__(self, feats: pd.DataFrame, ref_texts: list[str]):
        from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer

        self.feats = feats.drop_duplicates("text_hash").set_index("text_hash")
        cols = list(self.feats.columns)
        ref = self.feats.loc[[text_hash(t) for t in ref_texts]]
        self.mu, self.sd = ref.mean(), ref.std().replace(0, np.nan)
        self.blocks = {
            "style_surface": SURFACE,
            "style_punctuation": [c for c in cols if c.startswith("punct_")],
            "style_function_words": [c for c in cols if c.startswith("fw_")],
            "style_syntax_pos": [c for c in cols if c.startswith("pos_")],
        }
        self.dep = [c for c in cols if c.startswith("deplab_")]
        self.mfw = [c for c in cols if c.startswith("mfw_")]
        assert self.dep and self.mfw, "style features predate the v2 standard; re-run scripts/style_features.py"
        self.charwb = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=3, sublinear_tf=True).fit(ref_texts)
        self.char3 = HashingVectorizer(analyzer="char", ngram_range=(3, 3), n_features=2**22, alternate_sign=False,
                                       norm="l2", preprocessor=lambda t: re.sub(r"\s+", " ", t.lower()))

    def score(self, ref: pd.Series, hyp: pd.Series, chunk: int = 20_000) -> pd.DataFrame:
        parts = []
        for s in range(0, len(ref), chunk):
            r, h = ref.iloc[s:s + chunk], hyp.iloc[s:s + chunk]
            fr = self.feats.loc[[text_hash(t) for t in r]]
            fh = self.feats.loc[[text_hash(t) for t in h]]
            z = lambda f, c: ((f[c] - self.mu[c]) / self.sd[c]).fillna(0).to_numpy()
            out = {k: _rowcos(z(fr, c), z(fh, c)) for k, c in self.blocks.items()}
            pu = self.blocks["style_punctuation"]
            out["punct_l1"] = np.abs(fr[pu].to_numpy() - fh[pu].to_numpy()).sum(1)
            out["fw_cosine"] = _rowcos(fr[self.blocks["style_function_words"]].to_numpy(),
                                       fh[self.blocks["style_function_words"]].to_numpy())
            out["pos_jsd"] = _rowjsd(fr[self.blocks["style_syntax_pos"]].to_numpy(),
                                     fh[self.blocks["style_syntax_pos"]].to_numpy())
            out["dep_jsd"] = _rowjsd(fr[self.dep].to_numpy(), fh[self.dep].to_numpy())
            zr = ((fr[self.mfw] - self.mu[self.mfw]) / self.sd[self.mfw]).to_numpy()
            zh = ((fh[self.mfw] - self.mu[self.mfw]) / self.sd[self.mfw]).to_numpy()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out["burrows_delta"] = np.nanmean(np.abs(zr - zh), axis=1)
            dot = lambda v: np.asarray(v.transform(r).multiply(v.transform(h)).sum(axis=1)).ravel()
            out["style_charngram"], out["char3_cosine"] = dot(self.charwb), dot(self.char3)
            parts.append(pd.DataFrame(out))
        return pd.concat(parts, ignore_index=True)[STYLE_COMPONENTS]


# -- retained scale and floors -------------------------------------------------------------------

def retained(score, floor, metric: str):
    """1 = identical to T0, 0 = no closer than an unrelated T0 text of the same dataset.
    Similarities: (s - floor) / (1 - floor). Distances (DISTANCES): (floor - s) / floor."""
    return (floor - score) / floor if metric in DISTANCES else (score - floor) / (1 - floor)


def add_style_composite(df: pd.DataFrame) -> pd.DataFrame:
    """Family scores (style_fam_*) and the combined `style_ret` from the components' *_ret columns."""
    for fam, comps in STYLE_FAMILIES.items():
        df[f"style_fam_{fam}_ret"] = df[[f"{c}_ret" for c in comps]].mean(axis=1)
    df["style_ret"] = df[[f"style_fam_{f}_ret" for f in STYLE_FAMILIES]].mean(axis=1)
    return df


def t0_floor_pairs(corpus: pd.DataFrame, n_per_dataset: int = 200, seed: int = 6120) -> pd.DataFrame:
    """Unrelated T0 pairs: two originals of the same dataset from different keys (so different articles)."""
    rng = np.random.default_rng([seed, 32])
    rows = []
    for ds, part in corpus[corpus["round"] == 0].groupby("dataset"):
        part, n = part.reset_index(drop=True), 0
        while n < n_per_dataset:
            i, j = rng.choice(len(part), 2, replace=False)
            if part.key[i] != part.key[j]:
                rows.append({"dataset": ds, "ref": part.text[i], "hyp": part.text[j]})
                n += 1
    return pd.DataFrame(rows).assign(floor_id=lambda d: range(len(d)))


# -- Samer's extra metrics (ported from sot_research, same definitions) ----------------------------

def extra_lexical_scores(ref: str, hyp: str) -> dict:
    """chrF (sacrebleu CHRF defaults: char order 6, beta 2) / 100; word edit rate = word-level Levenshtein
    distance over whitespace tokens / ref words (can exceed 1); exact equality of the stripped texts."""
    import sacrebleu
    from rapidfuzz.distance import Levenshtein

    r, h = ref.split(), hyp.split()
    return {"chrf": sacrebleu.sentence_chrf(hyp, [ref]).score / 100,
            "word_edit_rate": Levenshtein.distance(r, h) / max(len(r), 1),
            "exact_equal": float(ref.strip() == hyp.strip())}


ENT_LABELS = {"PERSON", "ORG", "GPE", "LOC", "NORP", "FAC", "EVENT", "WORK_OF_ART", "LAW", "PRODUCT"}
UNITS = {"%", "percent", "per", "million", "billion", "trillion", "thousand", "hundred", "m", "km", "kg", "g",
         "mph", "kph", "km/h", "ft", "feet", "miles", "mile", "metres", "meters", "metre", "meter", "years", "year",
         "months", "month", "weeks", "week", "days", "day", "hours", "hour", "minutes", "minute", "seconds",
         "pounds", "dollars", "euros", "people", "tonnes", "tons", "cm", "mm", "litres", "liters", "bn", "mn", "pm",
         "am", "st", "nd", "rd", "th"}
CURRENCY = {"£": "gbp", "$": "usd", "€": "eur"}
NUMBER = re.compile(r"(?<![\w.])([£$€]?)(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(?![\d])")
COVERAGE_THRESHOLD = 0.6  # an original sentence counts as covered when its best MPNet match reaches this


def numbers(text: str) -> tuple[list, list]:
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


def extract_info(texts: list[str], n_process: int = 1) -> pd.DataFrame:
    """Named entities, numbers, quantities and spaCy sentences per text (keyed by text_hash).
    Use n_process > 1 only from a script with a __main__ guard (it deadlocks inside Jupyter on macOS)."""
    import json

    import spacy

    nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
    nlp.max_length = 2_000_000
    rows = []
    for doc in nlp.pipe(texts, batch_size=64, n_process=n_process):
        ents = sorted({(norm_entity(e.text), e.label_) for e in doc.ents if e.label_ in ENT_LABELS})
        nums, qty = numbers(doc.text)
        sents = [s.text.strip() for s in doc.sents if any(not t.is_space for t in s)]
        rows.append({"entities_json": json.dumps([e for e in ents if e[0]]), "numbers_json": json.dumps(nums),
                     "quantities_json": json.dumps(qty), "sentences_json": json.dumps(sents)})
    return pd.DataFrame(rows).assign(text_hash=[text_hash(t) for t in texts])


def information_scores(ref_info: dict, hyp: str, hyp_info: dict) -> dict:
    """Entity recall (normalized ref entities found verbatim in the hyp), number and quantity recall (clipped
    multisets), numbers added. HEURISTIC warnings, not validated factual errors (Samer's definitions)."""
    import json
    from collections import Counter

    ents = [e for e, _ in json.loads(ref_info["entities_json"])]
    low = hyp.lower()
    lost = sum(re.search(r"(?<!\w)" + re.escape(e) + r"(?!\w)", low) is None for e in ents)
    rn, hn = Counter(json.loads(ref_info["numbers_json"])), Counter(json.loads(hyp_info["numbers_json"]))
    rq, hq = Counter(json.loads(ref_info["quantities_json"])), Counter(json.loads(hyp_info["quantities_json"]))
    kept_n, kept_q = sum(min(c, hn[k]) for k, c in rn.items()), sum(min(c, hq[k]) for k, c in rq.items())
    return {"entities_ref": len(ents), "entities_lost": lost,
            "entity_recall": 1 - lost / len(ents) if ents else np.nan,
            "numbers_ref": sum(rn.values()), "numbers_lost": sum(rn.values()) - kept_n,
            "numbers_added": sum(hn.values()) - kept_n,
            "number_recall": kept_n / sum(rn.values()) if rn else np.nan,
            "quantity_recall": kept_q / sum(rq.values()) if rq else np.nan}


def _info_worker(args):
    ref_info, hyp, hyp_info = args
    return information_scores(ref_info, hyp, hyp_info)


def score_information_chunked(pairs: pd.DataFrame, info: pd.DataFrame, out_path: Path, key_cols: list[str],
                              workers: int = 6, chunk: int = 20_000) -> pd.DataFrame:
    """Information scores for every pair, chunked and resumable like score_lexical_chunked."""
    import time
    from concurrent.futures import ProcessPoolExecutor

    out_path = Path(out_path)
    parts_dir = out_path.with_suffix(".parts")
    parts_dir.mkdir(exist_ok=True)
    cols = ["entities_json", "numbers_json", "quantities_json"]
    by_hash = info.drop_duplicates("text_hash").set_index("text_hash")[cols]
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, start in enumerate(range(0, len(pairs), chunk)):
            part = parts_dir / f"part_{i:04d}.parquet"
            if part.exists():
                continue
            frame = pairs.iloc[start:start + chunk]
            ri = by_hash.loc[[text_hash(t) for t in frame.ref]].to_dict("records")
            hi = by_hash.loc[[text_hash(t) for t in frame.hyp]].to_dict("records")
            rows = list(pool.map(_info_worker, zip(ri, frame.hyp, hi), chunksize=200))
            pd.concat([frame[key_cols].reset_index(drop=True), pd.DataFrame(rows)], axis=1).to_parquet(part, index=False)
            print(f"information chunk {i + 1}/{-(-len(pairs) // chunk)} done, {time.time() - started:.0f}s", flush=True)
    out = pd.concat([pd.read_parquet(p) for p in sorted(parts_dir.glob("part_*.parquet"))], ignore_index=True)
    assert len(out) == len(pairs), "information parts don't cover all pairs"
    out.to_parquet(out_path, index=False)
    return out


def coverage(ref_vecs: np.ndarray, hyp_vecs: np.ndarray, threshold: float = COVERAGE_THRESHOLD) -> dict:
    """Best-match MPNet cosine of each ref sentence in the hyp (and vice versa)."""
    if not len(ref_vecs) or not len(hyp_vecs):
        return {"sent_coverage": np.nan, "min_sent_sim": np.nan, "sentences_uncovered": np.nan,
                "sentences_unsupported": np.nan, "ref_sentences": len(ref_vecs)}
    S = ref_vecs @ hyp_vecs.T
    best_r, best_h = S.max(1), S.max(0)
    return {"sent_coverage": float((best_r >= threshold).mean()), "min_sent_sim": float(best_r.min()),
            "sentences_uncovered": int((best_r < threshold).sum()),
            "sentences_unsupported": int((best_h < threshold).sum()), "ref_sentences": len(ref_vecs)}


def score_neural_extra(jobs, sentences_of: dict, cache_path: Path, id_cols: list[str], device: str = "auto",
                       dtype: str = "fp32", jobs_per_batch: int = 1, batch_size: int = 16, flush_every: int = 50,
                       log_every: int = 200) -> pd.DataFrame:
    """E5 document cosine (intfloat/e5-base-v2, "query: " prefix, 512-token sentence-packed chunks,
    token-weighted) and MPNet sentence coverage for every pair. `sentences_of`: text_hash -> spaCy sentences.
    Resumable like score_semantic."""
    import time

    cache_path = Path(cache_path)
    done = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame()
    done_ids = set(map(tuple, done[id_cols].drop_duplicates().to_numpy())) if len(done) else set()
    e5 = SentenceChunkEmbedder("intfloat/e5-base-v2", max_tokens=512, prefix="query: ", device=device, dtype=dtype,
                               batch_size=batch_size)
    mp = SentenceChunkEmbedder(device=device, dtype=dtype, batch_size=max(batch_size, 64))
    rows, n_jobs, started, group = [], 0, time.time(), []

    def flush():
        nonlocal done, rows
        if rows:
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
            tmp = cache_path.with_suffix(".tmp.parquet")
            done.to_parquet(tmp, index=False)
            tmp.replace(cache_path)
            rows = []

    def run(group):
        nonlocal n_jobs
        texts = [t for _, tx, _ in group for t in tx]
        ve = e5.embed(texts)
        sents = [sentences_of[text_hash(t)] for t in texts]
        uniq = list(dict.fromkeys(x for ss in sents for x in ss))
        sv = dict(zip(uniq, mp.embed(uniq))) if uniq else {}
        dim = mp.model.config.hidden_size
        sm = [np.array([sv[x] for x in ss]) if ss else np.zeros((0, dim)) for ss in sents]
        off = 0
        for meta, tx, prs in group:
            for r, h, pm in prs:
                rows.append({**meta, **pm, "e5_cosine": float(ve[off + h] @ ve[off + r]),
                             **coverage(sm[off + r], sm[off + h])})
            off += len(tx)
            n_jobs += 1
            if n_jobs % flush_every == 0:
                flush()
            if n_jobs % log_every == 0:
                print(f"{n_jobs} jobs, {time.time() - started:.0f}s", flush=True)

    for job in jobs:
        if tuple(job[0][c] for c in id_cols) in done_ids:
            continue
        group.append(job)
        if len(group) >= jobs_per_batch:
            run(group)
            group = []
    if group:
        run(group)
    flush()
    print(f"neural extra done: {n_jobs} new jobs, {len(done):,} rows, {time.time() - started:.0f}s; "
          f"chunked (E5): {e5.chunked_texts}", flush=True)
    return done
