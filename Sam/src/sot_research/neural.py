"""Neural semantic metrics without silent truncation, with embedding/score caches keyed by model revision.

E5 / MPNet document cosine: each text is split into sentences (regex (?<=[.!?])\\s+, the MT pipelines' splitter),
packed greedily into chunks that fit the encoder's window (E5 512 tokens incl. the "query: " prefix, MPNet 384),
each chunk mean-pooled and L2-normalized; the document vector is the token-count-weighted mean of its chunk
vectors, re-normalized. A text that fits one window gets exactly the standard single-pass embedding.

BERTScore: bert_score 0.3.13 conventions (roberta-large, layer 17, no idf, greedy matching; F1 reported raw and
baseline-rescaled with bert_score's own baseline file). Texts longer than 510 tokens are embedded in overlapping
windows (core 384 tokens, 63 tokens of context each side), following the team's LongBertScore
(src/pilot/metrics.py), so every token is matched and none is truncated.

Sentence coverage (information preservation, HEURISTIC): every ref sentence's best MPNet cosine against the hyp's
sentences; a ref sentence is "uncovered" below the configured threshold, a hyp sentence "unsupported" when no ref
sentence reaches it.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths

SPLIT = re.compile(r"(?<=[.!?])\s+")
NEURAL_VERSION = "neural-v1"


def best_device() -> str:
    """cuda if present, else cpu. Apple MPS is opt-in (SOT_RESEARCH_DEVICE=mps): torch 2.15's MPS backend aborts
    on macOS 14.3 (MPSNDArray selector error), and CPU results are the reference."""
    import os

    import torch

    if os.environ.get("SOT_RESEARCH_DEVICE"):
        return os.environ["SOT_RESEARCH_DEVICE"]
    return "cuda" if torch.cuda.is_available() else "cpu"


def snapshot(repo: str) -> tuple:
    """Local snapshot path and its commit hash (the revision actually used)."""
    from huggingface_hub import try_to_load_from_cache

    cfg = try_to_load_from_cache(repo, "config.json")
    if not isinstance(cfg, str):
        raise FileNotFoundError(f"{repo} is not in the local Hugging Face cache; download it first")
    p = Path(cfg).parent
    return p, p.name


def _fp(blob: dict) -> str:
    return hashlib.sha256(json.dumps(blob, sort_keys=True).encode()).hexdigest()[:12]


class EmbeddingCache:
    """sha256 -> vector, persisted as .npz (atomic replace)."""

    def __init__(self, path: Path):
        self.path = path
        self.data = {}
        if path.exists():
            z = np.load(path, allow_pickle=False)
            self.data = dict(zip(z["keys"].tolist(), z["emb"]))

    def save(self):
        if not self.data:
            return
        keys = np.array(list(self.data))
        tmp = self.path.with_name(self.path.stem + ".tmp.npz")
        np.savez(tmp, keys=keys, emb=np.stack([self.data[k] for k in keys]))
        tmp.replace(self.path)


class ChunkEmbedder:
    def __init__(self, repo: str, max_tokens: int, prefix: str = "", device: str | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.path, self.revision = snapshot(repo)
        self.repo, self.max_tokens, self.prefix = repo, max_tokens, prefix
        self.tok = AutoTokenizer.from_pretrained(self.path)
        self.device = device or best_device()
        self.model = AutoModel.from_pretrained(self.path).to(self.device).eval()
        self.n_prefix = len(self.tok(prefix, add_special_tokens=False)["input_ids"]) if prefix else 0
        self.chunked_texts = 0

    @property
    def fingerprint(self) -> str:
        return _fp({"v": NEURAL_VERSION, "repo": self.repo, "rev": self.revision, "max": self.max_tokens,
                    "prefix": self.prefix, "pool": "mean+l2", "chunk": "sentence-pack, token-weighted mean"})

    def _len(self, s: str) -> int:
        return len(self.tok(self.prefix + s, add_special_tokens=True)["input_ids"])

    def chunks(self, text: str) -> list:
        """Sentence-packed chunks that each fit the window (verified with the tokenizer)."""
        def fit(parts: list) -> list:
            s = " ".join(parts)
            if self._len(s) <= self.max_tokens:
                return [s]
            if len(parts) > 1:
                h = len(parts) // 2
                return fit(parts[:h]) + fit(parts[h:])
            words = s.split()
            if len(words) <= 1:
                raise ValueError("a single word exceeds the encoder window")
            h = len(words) // 2
            return fit([" ".join(words[:h])]) + fit([" ".join(words[h:])])
        sents = [x for x in SPLIT.split(text.strip()) if x] or [text.strip()]
        if self._len(" ".join(sents)) <= self.max_tokens:
            return [" ".join(sents)]
        # greedy packing, then exact verification/splitting
        out, cur = [], []
        for s in sents:
            if cur and self._len(" ".join(cur + [s])) > self.max_tokens:
                out += fit(cur)
                cur = []
            cur.append(s)
        if cur:
            out += fit(cur)
        return out

    def _encode(self, chunks: list, batch_size: int = 16) -> tuple:
        torch = self.torch
        vecs, lens = [], []
        for k in range(0, len(chunks), batch_size):
            b = self.tok([self.prefix + c for c in chunks[k:k + batch_size]], padding=True, truncation=False,
                         return_tensors="pt").to(self.device)
            if b["input_ids"].shape[1] > self.max_tokens:
                raise ValueError("chunk exceeds the window")       # never truncate silently
            with torch.no_grad():
                h = self.model(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1)
            vecs.append(torch.nn.functional.normalize(v, dim=-1).cpu().numpy())
            lens += b["attention_mask"].sum(1).tolist()
        return np.concatenate(vecs), np.array(lens, float)

    def embed(self, texts: list) -> np.ndarray:
        all_chunks, owner = [], []
        for i, t in enumerate(texts):
            c = self.chunks(t)
            self.chunked_texts += len(c) > 1
            all_chunks += c
            owner += [i] * len(c)
        v, n = self._encode(all_chunks)
        owner = np.array(owner)
        out = np.stack([(v[owner == i] * n[owner == i, None]).sum(0) for i in range(len(texts))])
        return (out / np.linalg.norm(out, axis=1, keepdims=True)).astype(np.float32)


def embed_cached(emb: ChunkEmbedder, items: dict, name: str, log=print, batch: int = 64) -> dict:
    """items: sha -> text. Returns sha -> vector, computing only missing ones."""
    cache = EmbeddingCache(paths.cache(f"emb__{name}__{emb.fingerprint}.npz"))
    todo = [k for k in items if k not in cache.data]
    log(f"{name} embeddings [{emb.fingerprint} rev {emb.revision[:10]}]: {len(items)} needed, "
        f"{len(items) - len(todo)} cached, {len(todo)} to embed")
    for k in range(0, len(todo), batch):
        keys = todo[k:k + batch]
        for key, v in zip(keys, emb.embed([items[x] for x in keys])):
            cache.data[key] = v
        if (k // batch) % 10 == 9 or k + batch >= len(todo):
            cache.save()
            log(f"  {min(k + batch, len(todo))}/{len(todo)}")
    return {k: cache.data[k] for k in items}


class LongBertScore:
    def __init__(self, repo: str = "FacebookAI/roberta-large", layer: int = 17, window: int = 510, step: int = 384,
                 device: str | None = None):
        import bert_score
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.path, self.revision = snapshot(repo)
        self.repo, self.layer = repo, layer
        self.device = device or best_device()
        self.tok = AutoTokenizer.from_pretrained(self.path)
        model = AutoModel.from_pretrained(self.path)
        model.encoder.layer = model.encoder.layer[:layer]     # bert_score truncates the encoder at `layer`
        self.model = model.to(self.device).eval()
        self.window, self.step, self.margin = window, step, (window - step) // 2
        base = Path(bert_score.__file__).parent / "rescale_baseline/en/roberta-large.tsv"
        row = pd.read_csv(base).set_index("LAYER").loc[layer]
        self.baseline = {"P": float(row["P"]), "R": float(row["R"]), "F": float(row["F"])}
        self.bert_score_version = bert_score.__version__
        self.windowed_texts = 0

    @property
    def fingerprint(self) -> str:
        return _fp({"v": NEURAL_VERSION, "repo": self.repo, "rev": self.revision, "layer": self.layer,
                    "window": self.window, "step": self.step, "idf": False, "baseline": self.baseline})

    def _windows(self, n: int) -> list:
        if n <= self.window:
            return [(0, n, 0, n)]
        out = []
        for c in range(0, n, self.step):
            d = min(c + self.step, n)
            out.append((max(0, c - self.margin), min(n, d + self.margin), c, d))
        return out

    def embed(self, texts: list, batch_size: int = 8) -> list:
        torch = self.torch
        pieces = []
        ids_all = [self.tok(t.strip(), add_special_tokens=False, add_prefix_space=True)["input_ids"] for t in texts]
        for i, ids in enumerate(ids_all):
            wins = self._windows(len(ids))
            self.windowed_texts += len(wins) > 1
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
                parts[i].append((hi[0], hi[1 + c:1 + d], hi[n - 1]))
        out = []
        for i in range(len(texts)):
            first, last = parts[i][0][0], parts[i][-1][2]
            e = torch.cat([first[None], *[core for _, core, _ in parts[i]], last[None]])
            out.append(torch.nn.functional.normalize(e, dim=-1))
        return out

    def score(self, cand, ref) -> dict:
        sim = cand @ ref.T
        p = sim[1:-1].max(dim=1).values.mean().item()
        r = sim[:, 1:-1].max(dim=0).values.mean().item()
        f = 2 * p * r / (p + r) if p + r else 0.0
        b = self.baseline
        return {"bertscore_p": p, "bertscore_r": r, "bertscore_f1": f,
                "bertscore_f1_rescaled": (f - b["F"]) / (1 - b["F"])}


def bertscore_pairs(pairs: pd.DataFrame, text_of: dict, bs: LongBertScore, log=print) -> pd.DataFrame:
    """BERTScore for each unique (ref, hyp) sha pair, resuming from a parquet cache. T0 embeddings stay in memory."""
    key = ["ref_sha256", "hyp_sha256"]
    cache = paths.cache(f"bertscore__{bs.fingerprint}.parquet")
    done = pd.read_parquet(cache) if cache.exists() else pd.DataFrame(columns=key)
    have = set(map(tuple, done[key].to_numpy()))
    todo = [tuple(x) for x in pairs.drop_duplicates(key)[key].to_numpy() if tuple(x) not in have]
    log(f"BERTScore [{bs.fingerprint} rev {bs.revision[:10]}]: {len(have)} cached, {len(todo)} to score")
    mem, rows = {}, []
    use = pd.Series([r for r, _ in todo] + [h for _, h in todo]).value_counts().to_dict()

    def get(sha):
        if sha not in mem:
            mem[sha] = bs.embed([text_of[sha]])[0].half()
        return mem[sha]
    for k, (r, h) in enumerate(todo):
        rows.append({"ref_sha256": r, "hyp_sha256": h, **bs.score(get(h).float(), get(r).float())})
        for s in (r, h):
            use[s] -= 1
            if use[s] == 0:
                mem.pop(s, None)
        if (k + 1) % 200 == 0 or k + 1 == len(todo):
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True) if len(done) else pd.DataFrame(rows)
            rows = []
            tmp = cache.with_suffix(".tmp")
            done.to_parquet(tmp, index=False)
            tmp.replace(cache)
            log(f"  {k + 1}/{len(todo)} (windowed texts so far: {bs.windowed_texts})")
    return done


def coverage(ref_sents: list, hyp_sents: list, sent_vec: dict, threshold: float) -> dict:
    """Best-match cosine of each ref sentence in the hyp (and vice versa)."""
    from .corpus import sha256_text

    if not ref_sents or not hyp_sents:
        return {"sent_coverage": np.nan, "min_sent_sim": np.nan, "sentences_uncovered": np.nan,
                "sentences_unsupported": np.nan, "ref_sentences": len(ref_sents)}
    R = np.stack([sent_vec[sha256_text(s)] for s in ref_sents])
    H = np.stack([sent_vec[sha256_text(s)] for s in hyp_sents])
    S = R @ H.T
    best_r, best_h = S.max(1), S.max(0)
    return {"sent_coverage": float((best_r >= threshold).mean()), "min_sent_sim": float(best_r.min()),
            "sentences_uncovered": int((best_r < threshold).sum()),
            "sentences_unsupported": int((best_h < threshold).sum()), "ref_sentences": len(ref_sents)}
