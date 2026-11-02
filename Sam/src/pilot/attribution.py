"""Source attribution (RQ2): can a classifier trained on original T0 texts still name a text's source
(Human or one of six LLMs) after it has been rewritten?

Ported from Samer Ahmed's `sot_research.classify` and `sot_research.features` (branch feature/sam-experiments)
with identical definitions, so paraphrase results can be pooled with his round-trip translation results:
- models and fixed hyperparameters (`make_model`), article folds (`article_folds`), grouped cross-validation
  (`grouped_cv`), scoring with an article-cluster bootstrap (`scores`), per-class scores and confusion matrices;
- the shared-opening audit and ablation (`shared_opening_lengths`, `strip_opening`);
- his per-text stylometric features (`stylometric_features`, `style_columns`), computed with HIS reference lists
  (function words / most frequent words fitted on the 200 XSum T0 texts). These are not the team-standard
  `style_features`; they exist to reproduce his stylometric classifier exactly.

Leakage rule: folds are formed over ORIGINAL ARTICLE IDS, so every text derived from an article (all sources, every
rewriter, every round) is on the same side of a split. Training uses T0 texts of the training articles only.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd

LENGTH_COLS = ["log_words", "log_sentences", "sent_len_mean", "word_len_mean"]
MODELS = ["tfidf_word", "tfidf_char", "stylometric", "length_only"]


# -- models and grouped cross-validation ---------------------------------------------------------------------

def make_model(name: str):
    """('text' | 'features', estimator). Fixed hyperparameters, never tuned."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if name == "tfidf_word":
        return "text", Pipeline([("v", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                                 ("clf", LogisticRegression(C=10, max_iter=5000, class_weight="balanced"))])
    if name == "tfidf_char":
        return "text", Pipeline([("v", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                                       sublinear_tf=True, max_features=200_000)),
                                 ("clf", LogisticRegression(C=10, max_iter=5000, class_weight="balanced"))])
    if name in ("stylometric", "length_only"):
        # median imputation fitted on the training fold: MATTR is undefined for texts under 2 words
        return "features", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()),
                                     ("clf", LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced"))])
    raise ValueError(name)


def add_length_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(log_words=np.log1p(df["n_words"]), log_sentences=np.log1p(df["n_sentences"]))


def article_folds(articles, n_splits: int, seed: int) -> dict:
    """Shuffle the unique article ids and deal them round-robin into n_splits folds."""
    u = np.array(sorted(set(articles)))
    u = u[np.random.default_rng(seed).permutation(len(u))]
    return {a: i % n_splits for i, a in enumerate(u)}


def grouped_cv(train_pool: pd.DataFrame, eval_sets: dict, label: str, model: str, cols=None, n_splits: int = 5,
               repeats: int = 5, seed: int = 6120,
               keep=("text_id", "item_id", "article_id", "source_label", "rewriter", "stage")) -> pd.DataFrame:
    """Train on `train_pool` rows of the training articles; predict every eval set's rows of the held-out articles.
    Returns one row per (repeat, eval set, text) with y_true and y_pred."""
    out = []
    articles = set(train_pool.article_id)
    for df in eval_sets.values():
        articles |= set(df.article_id)
    for r in range(repeats):
        folds = article_folds(articles, n_splits, seed + r)
        for k in range(n_splits):
            tr = train_pool[train_pool.article_id.map(folds) != k]
            kind, est = make_model(model)
            est.fit(tr["text"] if kind == "text" else tr[cols].to_numpy(float), tr[label].to_numpy())
            for name, df in eval_sets.items():
                te = df[df.article_id.map(folds) == k]
                if not len(te):
                    continue
                part = te[[c for c in keep if c in te]].copy()
                part["y_true"] = te[label].to_numpy()
                part["y_pred"] = est.predict(te["text"] if kind == "text" else te[cols].to_numpy(float))
                part["repeat"], part["fold"], part["eval_set"], part["model"] = r, k, name, model
                out.append(part)
    return pd.concat(out, ignore_index=True)


# -- scoring --------------------------------------------------------------------------------------------------

def macro_f1(yt, yp, labels) -> float:
    f = []
    for c in labels:
        tp, fp, fn = np.sum((yt == c) & (yp == c)), np.sum((yt != c) & (yp == c)), np.sum((yt == c) & (yp != c))
        f.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return float(np.mean(f))


def scores(preds: pd.DataFrame, by: list, labels: list, n_boot: int = 1000, seed: int = 6120) -> pd.DataFrame:
    """Accuracy and macro-F1 per group: mean and SD over repeats, and a 95% article-cluster bootstrap CI on the
    predictions pooled over repeats. n_texts = unique texts behind the score."""
    rng = np.random.default_rng(seed)
    labels = list(labels)
    rows = []
    for g, part in preds.groupby(by, observed=True):
        g = g if isinstance(g, tuple) else (g,)
        per_rep = part.groupby("repeat").apply(
            lambda d: pd.Series({"acc": float((d.y_true == d.y_pred).mean()),
                                 "f1": macro_f1(d.y_true.to_numpy(), d.y_pred.to_numpy(), labels)}),
            include_groups=False)
        arts = part.article_id.unique()
        idx = {a: np.flatnonzero(part.article_id.to_numpy() == a) for a in arts}
        yt, yp = part.y_true.to_numpy(), part.y_pred.to_numpy()
        boots_a, boots_f = [], []
        for _ in range(n_boot):
            sel = np.concatenate([idx[a] for a in rng.choice(arts, len(arts), replace=True)])
            boots_a.append((yt[sel] == yp[sel]).mean())
            boots_f.append(macro_f1(yt[sel], yp[sel], labels))
        rows.append({**dict(zip(by, g)), "accuracy": per_rep.acc.mean(), "accuracy_sd": per_rep.acc.std(ddof=1),
                     "accuracy_ci_low": np.percentile(boots_a, 2.5), "accuracy_ci_high": np.percentile(boots_a, 97.5),
                     "macro_f1": per_rep.f1.mean(), "macro_f1_sd": per_rep.f1.std(ddof=1),
                     "macro_f1_ci_low": np.percentile(boots_f, 2.5), "macro_f1_ci_high": np.percentile(boots_f, 97.5),
                     "n_texts": part.text_id.nunique(), "n_articles": len(arts), "repeats": part.repeat.nunique(),
                     "chance": 1 / len(labels)})
    return pd.DataFrame(rows)


def scores_fast(preds: pd.DataFrame, by: list, labels: list, n_boot: int = 1000, seed: int = 6120) -> pd.DataFrame:
    """Same statistics as `scores`, vectorized for the full corpus (millions of predictions). The bootstrap draws
    multinomial article weights and sums per-article confusion counts, which is equivalent to resampling articles
    with replacement (only the random stream differs from `scores`)."""
    rng = np.random.default_rng(seed)
    labels = list(labels)
    L = len(labels)
    code = {c: i for i, c in enumerate(labels)}
    rows = []
    for g, part in preds.groupby(by, observed=True):
        g = g if isinstance(g, tuple) else (g,)
        yt = part.y_true.map(code).to_numpy()
        yp = part.y_pred.map(code).to_numpy()
        art, a_idx = np.unique(part.article_id.to_numpy(), return_inverse=True)
        A = len(art)
        # per article: tp, fp, fn per class, correct, total
        C = np.zeros((A, 3 * L + 2))
        ok = yt == yp
        np.add.at(C, (a_idx, yt), ok)                    # tp
        np.add.at(C, (a_idx, L + yp), ~ok)               # fp
        np.add.at(C, (a_idx, 2 * L + yt), ~ok)           # fn
        np.add.at(C, (a_idx, 3 * L), ok)
        np.add.at(C, (a_idx, 3 * L + 1), 1)
        W = rng.multinomial(A, np.full(A, 1 / A), size=n_boot)
        S = W @ C
        tp, fp, fn = S[:, :L], S[:, L:2 * L], S[:, 2 * L:3 * L]
        den = 2 * tp + fp + fn
        f1 = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0).mean(axis=1)
        acc = S[:, 3 * L] / S[:, 3 * L + 1]
        rep = part.repeat.to_numpy()
        per_f, per_a = [], []
        for r in np.unique(rep):
            m = rep == r
            per_f.append(macro_f1(yt[m], yp[m], range(L)))
            per_a.append(float(ok[m].mean()))
        rows.append({**dict(zip(by, g)), "accuracy": np.mean(per_a), "accuracy_sd": np.std(per_a, ddof=1),
                     "accuracy_ci_low": np.percentile(acc, 2.5), "accuracy_ci_high": np.percentile(acc, 97.5),
                     "macro_f1": np.mean(per_f), "macro_f1_sd": np.std(per_f, ddof=1),
                     "macro_f1_ci_low": np.percentile(f1, 2.5), "macro_f1_ci_high": np.percentile(f1, 97.5),
                     "n_texts": part.text_id.nunique(), "n_articles": A, "repeats": len(per_f), "chance": 1 / L})
    return pd.DataFrame(rows)


def per_class(preds: pd.DataFrame, by: list, labels: list) -> pd.DataFrame:
    """Precision / recall / F1 per class, pooled over repeats."""
    rows = []
    for g, part in preds.groupby(by, observed=True):
        g = g if isinstance(g, tuple) else (g,)
        for c in labels:
            tp = ((part.y_true == c) & (part.y_pred == c)).sum()
            fp = ((part.y_true != c) & (part.y_pred == c)).sum()
            fn = ((part.y_true == c) & (part.y_pred != c)).sum()
            p, r = (tp / (tp + fp) if tp + fp else 0.0), (tp / (tp + fn) if tp + fn else 0.0)
            rows.append({**dict(zip(by, g)), "class": c, "precision": p, "recall": r,
                         "f1": 2 * p * r / (p + r) if p + r else 0.0,
                         "support_per_repeat": int((part.y_true == c).sum() / part.repeat.nunique())})
    return pd.DataFrame(rows)


def confusion(part: pd.DataFrame, labels: list, normalize: bool = True) -> pd.DataFrame:
    m = pd.crosstab(pd.Categorical(part.y_true, labels), pd.Categorical(part.y_pred, labels), dropna=False)
    m.index.name, m.columns.name = "true", "predicted"
    return m.div(m.sum(axis=1).replace(0, np.nan), axis=0) if normalize else m


# -- shared-opening audit and ablation -----------------------------------------------------------------------

def shared_opening_lengths(df: pd.DataFrame, group=("article_id", "rewriter", "stage")) -> pd.Series:
    """Words at the start of each text shared VERBATIM with another source's text of the same article, rewriter
    and stage (longest common whitespace-word prefix with any other text in the group). At T0 this recovers the
    ~30-word human opening the generating LLMs were given; in rewrites it is measured, never assumed."""
    out = pd.Series(0, index=df.index, dtype=int)
    for _, g in df.groupby(list(group)):
        toks = {i: t.split() for i, t in zip(g.index, g.text)}
        for i in g.index:
            best = 0
            for j in g.index:
                if i == j:
                    continue
                n = 0
                for x, y in zip(toks[i], toks[j]):
                    if x != y:
                        break
                    n += 1
                best = max(best, n)
            out[i] = best
    return out


def strip_opening(df: pd.DataFrame, lengths: pd.Series) -> pd.DataFrame:
    df = df.copy()
    df["text"] = [" ".join(t.split()[n:]) for t, n in zip(df.text, lengths.reindex(df.index).fillna(0).astype(int))]
    return df


# -- Samer's per-text stylometric features (spaCy en_core_web_sm) --------------------------------------------

WORD = re.compile(r"\b\w+\b")
PUNCT = [".", ",", ";", ":", "!", "?", '"', "'", "(", ")", "-", "—", "–", "“", "”", "‘", "’"]
UPOS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT",
        "SCONJ", "SYM", "VERB", "X"]
STYLE_SCALARS = ["sent_len_mean", "sent_len_sd", "word_len_mean", "mattr", "ttr", "cap_rate", "dep_dist_mean",
                 "dep_depth_mean"]


def _mattr(tokens: list, window: int = 50) -> float:
    if len(tokens) < 2:
        return np.nan
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    return float(np.mean([len(set(tokens[i:i + window])) / window for i in range(len(tokens) - window + 1)]))


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


def _doc_features(doc, ref: dict, deps: list) -> dict:
    toks = [t for t in doc if not t.is_space]
    n = max(len(toks), 1)
    text = doc.text
    words = [w.lower() for w in WORD.findall(text)]
    nw = max(len(words), 1)
    alpha = [t.text.lower() for t in toks if t.is_alpha]
    sents = [s for s in doc.sents if any(not t.is_space for t in s)]
    sent_lens = [sum(1 for t in s if not t.is_space and not t.is_punct) for s in sents] or [0]
    depth = _depths(doc)
    f = {"n_words": len(words), "n_sentences": len(sents),
         "sent_len_mean": float(np.mean(sent_lens)), "sent_len_sd": float(np.std(sent_lens)),
         "word_len_mean": float(np.mean([len(w) for w in alpha])) if alpha else 0.0,
         "mattr": _mattr(alpha), "ttr": len(set(alpha)) / max(len(alpha), 1),
         "cap_rate": sum(t.text[:1].isupper() for t in toks if t.is_alpha) / max(len(alpha), 1),
         "dep_dist_mean": float(np.mean([abs(t.i - t.head.i) for t in toks])) if toks else 0.0,
         "dep_depth_mean": float(np.mean(depth[[t.i for t in toks]])) if toks else 0.0}
    for i, p in enumerate(PUNCT):
        f[f"punct_{i:02d}"] = text.count(p) / n
    wc = Counter(words)
    for w in ref["function_words"]:
        f[f"fw_{w}"] = wc.get(w, 0) / nw
    pos = Counter(t.pos_ for t in toks)
    for p in UPOS:
        f[f"pos_{p}"] = pos.get(p, 0) / n
    dep = Counter(t.dep_ for t in toks)
    for d in deps:
        f[f"dep_{d}"] = dep.get(d, 0) / n
    return f


def stylometric_features(texts: list[str], ref: dict, n_process: int = 1) -> pd.DataFrame:
    """Samer's per-text features for each text (row order kept). n_process > 1 only from a script with a
    __main__ guard (spaCy multiprocessing deadlocks inside Jupyter on macOS)."""
    import spacy

    nlp = spacy.load("en_core_web_sm", disable=["lemmatizer"])
    nlp.max_length = 2_000_000
    deps = sorted(nlp.get_pipe("parser").labels)
    return pd.DataFrame([_doc_features(d, ref, deps) for d in nlp.pipe(texts, batch_size=32, n_process=n_process)])


def style_columns(df: pd.DataFrame) -> list:
    """Stylometric columns by allowlist: surface scalars and punctuation, function-word, UPOS and dependency rates."""
    pref = ("punct_", "fw_", "pos_", "dep_")
    return [c for c in df.columns if c in STYLE_SCALARS or
            (c.startswith(pref) and c not in ("dep_dist_mean", "dep_depth_mean", "dep_depth_max"))]
